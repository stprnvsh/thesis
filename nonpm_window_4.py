#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marked multivariate Hawkes with nonparametric temporal & spatial kernels,
and a finite time window W so each event only depends on the last W seconds.

IDENTIFIABILITY-FIXED VERSION (+ EVAL FIX, + SOE Quadratic, + Nonlinear link):
- Temporal kernel shape g̃ has unit integral (∫ g̃ = 1); spatial kernel κ̃ is
  normalized per source column across REACHABLE targets, with a selectable
  normalization:
    --spatial_norm mean   : col-mean(κ̃[:, j]) = 1  (default, previous behavior)
    --spatial_norm sum    : col-sum(κ̃[:, j])  = 1  (discrete analog of integral-1)
  A single scalar amplitude α carries total excitation scale.

- g̃(τ) = sum_k mix_w_k * N(τ; c_k, ℓ_t^2), τ >= 0, with ∫ g̃ = 1
- κ̃(r) uses a Gaussian-basis mixture with the selected column normalization.

- Quadratic kernel (SOE): z(τ) = Σ_{r=1..R} w_r β_r e^{-β_r τ} · 1{τ≥0},  Σ w_r = 1 (unit integral).
  Here R is set by --B_q (for CLI compatibility). --q_scale sets a LogNormal prior
  center for β roughly around 1/q_scale.

- Effective linear predictor η = μ + α Σ G∘κ̃ * g̃ + γ * (Σ G * z · sign)^2
  Intensity is λ = φ(η) with a selectable nonlinearity φ:
     --nonlinearity linear|softplus|relu|exp|power2
  The event term uses log λ. The compensator is:
    * analytic (base + excitation + SOE pair-integral) for φ = linear
    * numeric trapezoid on a grid for φ ≠ linear (windowed)

- Windowed likelihood: only pairs with (t_i - t_j) <= W contribute.
  The linear compensator uses α * ∫_0^{min(T - t_j, W)} g̃(τ) dτ.

Reporting / Evaluation FIX:
- Diagnostics compare “like with like” by normalizing the TRUE spatial kernel
  with the SAME per-column rule (mean or sum) used for κ̃.
- Also reports a best scalar c* for α·K∘κ̃ vs K_true∘κ_true to separate
  shape from scale.

Saves:
- Posterior means to inference_result_np_<data>.pickle
- Full posterior draws to mcmc_state_np_<data>.npz (for mcmc)
"""

import argparse
import pickle
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erf

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
from numpyro import enable_x64
from numpyro.infer.initialization import init_to_value

# ---------------- Platform (CPU-safe defaults) ----------------
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(10)  # optional


# ---------------- Utilities ----------------
def compute_reachability(adjacency, num_hops=1):
    A = (adjacency > 0).astype(np.int32)
    N = A.shape[0]
    R = np.eye(N, dtype=np.int32)
    cur = A.copy()
    for _ in range(num_hops):
        R = (R | (cur > 0).astype(np.int32)).astype(np.int32)
        cur = (cur @ A > 0).astype(np.int32)
    return R.astype(np.float32)


def prep_events_structured(events, num_event_types=None):
    t = np.asarray(events["t"])
    u = np.asarray(events["u"])
    e = np.asarray(events["e"])
    T = float(t.max()) if t.size > 0 else 0.0
    N = int(u.max()) + 1 if u.size > 0 else 0
    M = int(num_event_types) if num_event_types is not None else (int(e.max()) + 1 if e.size > 0 else 1)
    return t, u, e, T, N, M


def reconstruct_true_params_if_present(params_init, N, M):
    """
    params_init: [mu.flatten(), K.flatten(), omega(true), sigma(true)] from generator.
    Returns (mu_true, K_true, omega_true, sigma_true) or Nones.
    """
    if params_init is None:
        return None, None, None, None
    expected_len = N * M + N * N + 2
    if params_init.size != expected_len:
        return None, None, None, None
    mu_flat = params_init[: N * M]
    K_flat = params_init[N * M : N * M + N * N]
    omega_true = float(params_init[-2])
    sigma_true = float(params_init[-1])
    mu_true = mu_flat.reshape(N, M)
    K_true = K_flat.reshape(N, N)
    return mu_true, K_true, omega_true, sigma_true


def pairwise_dists(node_xy):  # (N,2) -> (N,N)
    diff = node_xy[:, None, :] - node_xy[None, :, :]
    d2 = jnp.sum(diff * diff, axis=-1)
    return jnp.sqrt(jnp.maximum(d2, 0.0))


def pairwise_sq_dists(node_xy):
    diff = node_xy[:, None, :] - node_xy[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


# ---------------- Gaussian basis utilities ----------------
def make_centers(width, n):
    if n == 1:
        return jnp.array([0.5 * width])
    return jnp.linspace(0.0, width, n)


def gauss_bump(x, c, scale):  # vectorized
    z = (x - c) / scale
    return jnp.exp(-0.5 * z * z)


def gauss_bump_int_0_to(x, c, scale):
    """
    ∫_0^x exp(-0.5 * ((t - c)/s)^2) dt
      = s * sqrt(pi/2) * [erf((x - c)/(sqrt(2)*s)) - erf((-c) / (sqrt(2)*s))]
    """
    rt2 = jnp.sqrt(2.0)
    pref = scale * jnp.sqrt(jnp.pi / 2.0)
    return pref * (erf((x - c) / (rt2 * scale)) - erf((-c) / (rt2 * scale)))


def gauss_bump_int_0_to_inf(c, scale):
    """
    ∫_0^∞ exp(-0.5 * ((t - c)/s)^2) dt
      = s * sqrt(pi/2) * [1 - erf((-c)/(sqrt(2)*s))]
    """
    rt2 = jnp.sqrt(2.0)
    return scale * jnp.sqrt(jnp.pi / 2.0) * (1.0 - erf((-c) / (rt2 * scale)))


# ---------------- Hawkes model with static-bounds window ----------------
def hawkes_np_model(
    t, u, e, T,
    node_xy, reach_mask,
    time_centers, time_scale,
    space_centers, space_scale,
    time_centers_q, q_scale,    # time_centers_q length (=R) used to set #exponentials
    start_idx,                  # (K,) int32/64 array, precomputed on host
    L_max,                      # scalar int: max(i - start_idx[i])
    W,                          # scalar (float) window; may be inf
    N: int, M: int,
    spatial_norm: str = "sum",  # "mean" or "sum"
    nonlinearity: str = "linear",
    grid_times: jnp.ndarray = None,
    grid_starts: jnp.ndarray = None,
    grid_ends: jnp.ndarray = None,
    Lg_max: int = 0,
):
    """
    Nonparametric temporal g̃(τ) (unit integral) and spatial κ̃(r) with per-column
    normalization over REACHABLE targets:
        spatial_norm="mean": col-mean(κ̃[:, j]) = 1
        spatial_norm="sum" : col-sum(κ̃[:, j])  = 1
    Effective kernel uses α * g̃ and κ̃. Uses fixed-length scan per event.

    Quadratic kernel uses a Sum-Of-Exponentials: z(τ)=Σ w_r β_r e^{-β_r τ}, τ>=0, Σw_r=1.
    Intensity uses λ = φ(η), where η = μ + excitation + γ(Σ G z sign)^2.
    """

    def phi_link(x):
        # numeric floor to keep log well-defined
        if nonlinearity == "linear":
            return jnp.clip(x, a_min=1e-12)
        elif nonlinearity == "softplus":
            return jax.nn.softplus(x) + 1e-12
        elif nonlinearity == "relu":
            return jnp.clip(x, a_min=0.0) + 1e-12
        elif nonlinearity == "exp":
            return jnp.exp(x) + 1e-12
        elif nonlinearity == "power2":
            z = jnp.clip(x, a_min=0.0)
            return z * z + 1e-12
        else:
            return jnp.clip(x, a_min=1e-12)

    Kevents = t.shape[0]

    # ---- Base rates and coupling matrices (positive via softplus)
    mu_uncon = numpyro.sample("mu_uncon", dist.Normal(0.0, 1.0).expand([N, M]).to_event(2))
    mu = numpyro.deterministic("mu", jax.nn.softplus(mu_uncon) + 1e-8)

    # Slightly wider priors to avoid over-shrinking
    K_uncon = numpyro.sample("K_uncon", dist.Normal(0.0, 1.2).expand([N, N]).to_event(2))
    K_pos = jax.nn.softplus(K_uncon)
    K_pre = K_pos * reach_mask
    colsum_K = jnp.maximum(jnp.sum(K_pre, axis=0), 1e-12)
    K_masked = numpyro.deterministic("K_masked", K_pre / colsum_K[None, :])

    M_uncon = numpyro.sample("M_uncon", dist.Normal(0.0, 1.2).expand([M, M]).to_event(2))
    M_pos = jax.nn.softplus(M_uncon) + 1e-8
    rowsum_M = jnp.maximum(jnp.sum(M_pos, axis=1), 1e-12)
    M_K = numpyro.deterministic("M_K", M_pos / rowsum_M[:, None])

    # ---- Global amplitude α (total excitation scale)
    alpha = numpyro.sample("alpha", dist.Beta(2.0, 4.0))  # mean ~ 0.33

    # ---- Spatial kernel κ̃(r) via Gaussian basis
    B_s = space_centers.shape[0]
    b_uncon = numpyro.sample("b_uncon", dist.Normal(0.0, 0.8).expand([B_s]).to_event(1))
    v_pos = jax.nn.softplus(b_uncon) + 1e-8

    D = pairwise_dists(node_xy)  # (N,N)
    psi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)  # (N,N,B_s)
    raw_kappa = jnp.tensordot(psi, v_pos, axes=[-1, 0])  # (N,N)

    # Normalize per source column over REACHABLE targets
    mask = reach_mask  # (N,N) in {0,1}
    col_counts = jnp.sum(mask, axis=0)  # (N,)
    col_counts = jnp.maximum(col_counts, 1.0)

    if spatial_norm == "sum":
        Z_s_col = jnp.sum(raw_kappa * mask, axis=0)               # sum over reachable targets
        Z_s_col = jnp.maximum(Z_s_col, 1e-12)
    else:  # "mean"
        Z_s_col = jnp.sum(raw_kappa * mask, axis=0) / col_counts  # mean over reachable targets
        Z_s_col = jnp.where(jnp.isfinite(Z_s_col), Z_s_col, 1.0)

    kappa_tilde = numpyro.deterministic("kappa_tilde", raw_kappa / Z_s_col[None, :])  # (N,N)
    G_node = numpyro.deterministic("G_node", K_masked * kappa_tilde)

    # ---- Temporal kernel g̃(τ) with unit integral on [0, ∞)
    B_t = time_centers.shape[0]
    a_uncon = numpyro.sample("a_uncon", dist.Normal(0.0, 0.8).expand([B_t]).to_event(1))
    w_pos = jax.nn.softplus(a_uncon) + 1e-8
    ints = jnp.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers])  # (B_t,)
    Z_t = jnp.dot(w_pos, ints) + 1e-12
    mix_w = w_pos / Z_t
    numpyro.deterministic("mix_w", mix_w)

    def g_tilde_scalar(delta):  # scalar Δ >= 0
        delta = jnp.maximum(delta, 0.0)
        phi = jnp.exp(-0.5 * ((delta - time_centers) / time_scale) ** 2)  # (B_t,)
        return jnp.dot(phi, mix_w)

    def G_tilde_int_vec(delta):   # vector Δ -> ∫_0^Δ g̃(τ) dτ
        delta = jnp.clip(delta, a_min=0.0)
        Phi_int = jnp.stack([gauss_bump_int_0_to(delta, c, time_scale) for c in time_centers], axis=-1)  # (K,B_t)
        return Phi_int @ mix_w  # (K,)

    # ---- Quadratic Hawkes additions (ZHawkes) — Sum of Exponentials kernel
    # We interpret B_q := R (# exponentials); time_centers_q values are unused.
    R = int(time_centers_q.shape[0])
    if R > 0:
        # weights on simplex -> unit-integral kernel z(τ) = Σ w_r β_r e^{-β_r τ}, τ>=0, Σ w_r = 1
        wq_uncon = numpyro.sample("wq_uncon", dist.Normal(0.0, 0.8).expand([R]).to_event(1))
        wq_pos = jax.nn.softplus(wq_uncon) + 1e-8
        w_q = numpyro.deterministic("w_q", wq_pos / (jnp.sum(wq_pos) + 1e-12))  # (R,)

        # positive rates β_r; center prior roughly at 1/q_scale (if provided)
        beta_loc = jnp.log(jnp.maximum(1.0 / (q_scale + 1e-12), 1e-3))
        beta_q = numpyro.sample("beta_q", dist.LogNormal(beta_loc, 0.75).expand([R]).to_event(1))  # (R,)
        beta_q = numpyro.deterministic("beta_q_det", beta_q)

        gamma = numpyro.sample("gamma", dist.HalfNormal(1.0))

        # Learnable polarity per mark in (-1, 1)
        signs_raw = numpyro.sample("signs_raw", dist.Normal(0.0, 1.0).expand([M]).to_event(1))
        signs = numpyro.deterministic("signs", jnp.tanh(signs_raw))

        # Gram of G_node columns for Q_{jk}
        GG = jnp.matmul(G_node.T, G_node)  # (N,N)

        # Precompute constants for pair integrals
        wb = w_q * beta_q                             # (R,)
        denom = beta_q[:, None] + beta_q[None, :]     # (R,R)
        C = (wb[:, None] * wb[None, :]) / (denom + 1e-12)  # (R,R)

        def z_scalar(delta):
            delta = jnp.maximum(delta, 0.0)
            return jnp.sum(wb * jnp.exp(-beta_q * delta))   # Σ w_r β_r e^{-β_r Δ}

        def iz_pair_exp(tj, tk, L, U):
            # ∫_L^U z(t - tj) z(t - tk) dt with SOE kernel
            ok = (U > L)
            eL_j = jnp.exp(-beta_q * (L - tj))  # (R,)
            eL_k = jnp.exp(-beta_q * (L - tk))  # (R,)
            eU_j = jnp.exp(-beta_q * (U - tj))  # (R,)
            eU_k = jnp.exp(-beta_q * (U - tk))  # (R,)
            EL = eL_j[:, None] * eL_k[None, :]  # (R,R)
            EU = eU_j[:, None] * eU_k[None, :]  # (R,R)
            val = jnp.sum(C * (EL - EU))
            return jnp.where(ok, val, jnp.array(0.0, dtype=t.dtype))
    else:
        gamma = jnp.array(0.0, dtype=t.dtype)
        signs = jnp.ones((M,), dtype=t.dtype)

        def z_scalar(delta):
            return jnp.array(0.0, dtype=t.dtype)

        def iz_pair_exp(tj, tk, L, U):
            return jnp.array(0.0, dtype=t.dtype)

    # ---- Event log-likelihood with fixed-length scan per event
    def step_event(carry, i):
        t_i = t[i]
        u_i = u[i]
        e_i = e[i]
        start_i = start_idx[i]

        def body(acc, k):
            j = i - 1 - k
            valid = (j >= start_i) & (j >= 0)
            j_clamped = jnp.clip(j, 0, Kevents - 1)
            dt = t_i - t[j_clamped]
            valid = valid & (dt <= W)
            g_val = g_tilde_scalar(dt)
            contrib = G_node[u_i, u[j_clamped]] * M_K[e[j_clamped], e_i] * (alpha * g_val)
            contrib = jnp.where(valid, contrib, jnp.array(0.0, dtype=t.dtype))
            return acc + contrib, None

        excite_sum, _ = lax.scan(body, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(L_max))

        if R > 0:
            def body_z(acc, k):
                j = i - 1 - k
                valid = (j >= start_i) & (j >= 0)
                j_clamped = jnp.clip(j, 0, Kevents - 1)
                dt = t_i - t[j_clamped]
                valid = valid & (dt <= W)
                z_val = z_scalar(dt)  # SOE
                zc = G_node[u_i, u[j_clamped]] * z_val * signs[e[j_clamped]]
                zc = jnp.where(valid, zc, jnp.array(0.0, dtype=t.dtype))
                return acc + zc, None
            z_sum, _ = lax.scan(body_z, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(L_max))
        else:
            z_sum = jnp.array(0.0, dtype=t.dtype)

        eta_ie = mu[u_i, e_i] + excite_sum + gamma * (z_sum * z_sum)
        lam_ie = phi_link(eta_ie)
        return carry + jnp.log(lam_ie), None

    event_loglik, _ = lax.scan(step_event, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))

    # ---- Compensator
    if nonlinearity == "linear":
        # analytic linear compensator (base + excitation) + SOE quadratic integral
        base_int = T * jnp.sum(mu)
        colsum_G = jnp.sum(G_node, axis=0)  # (N,)
        rowsum_MK = jnp.sum(M_K, axis=1)    # (M,)
        tail_limit = jnp.minimum(T - t, W)  # (K,)
        tail = alpha * G_tilde_int_vec(tail_limit)  # (K,)
        exc_int = jnp.sum(colsum_G[u] * rowsum_MK[e] * tail)

        if R > 0:
            def step_pairs(carry, j):
                tj = t[j]
                uj = u[j]
                ej = e[j]
                sj = signs[ej]
                start_j = start_idx[j]

                def inner(acc, kk):
                    k = j - kk - 1
                    valid = (k >= start_j) & (k >= 0)
                    k = jnp.clip(k, 0, Kevents - 1)
                    tk = t[k]
                    uk = u[k]
                    ek = e[k]
                    sk = signs[ek]
                    # time window overlap
                    L = jnp.maximum(tj, tk)
                    U = jnp.minimum(jnp.minimum(T, tj + W), tk + W)
                    ok = (U > L)
                    # pair integral (SOE)
                    iz = iz_pair_exp(tj, tk, L, U)
                    # Q_{jk} = s_j s_k GG[u_j, u_k]
                    qjk = sj * sk * GG[uj, uk]
                    # off-diagonal counted once here -> multiply by 2 to account symmetry
                    contrib = jnp.where(valid & ok, 2.0 * qjk * iz, jnp.array(0.0, dtype=t.dtype))
                    return acc + contrib, None

                # diagonal term j==j
                Ld = tj
                Ud = jnp.minimum(T, tj + W)
                iz_diag = iz_pair_exp(tj, tj, Ld, Ud)
                qjj = (sj * sj) * GG[uj, uj]
                acc0 = carry + qjj * iz_diag

                total, _ = lax.scan(inner, init=acc0, xs=jnp.arange(L_max))
                return total, None

            quad_sum, _ = lax.scan(step_pairs, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))
            quad_int = gamma * quad_sum
        else:
            quad_int = jnp.array(0.0, dtype=t.dtype)

        loglik = event_loglik - base_int - exc_int - quad_int

    else:
        # numeric compensator on a time grid for φ(η)
        def compensator_grid():
            Gg = grid_times.shape[0]
            if (Gg == 0) | (Lg_max == 0):
                # still need baseline if no events? baseline is inside φ(η) here; but
                # with empty grid this is 0, which is acceptable since φ integrates over [0,T]
                return jnp.array(0.0, dtype=t.dtype)

            def step_grid(carry, g):
                tg = grid_times[g]
                start_g = grid_starts[g]
                end_g   = grid_ends[g]

                excite_mat = jnp.zeros((N, M), dtype=t.dtype)  # α * sum outer products
                z_accum    = jnp.zeros((N,), dtype=t.dtype)    # Σ G[:, u_j] * sign[e_j] * z(Δ)

                def body_j(vals, k):
                    j = end_g - k
                    valid = (j >= start_g) & (j >= 0)
                    j = jnp.clip(j, 0, Kevents - 1)
                    dt = tg - t[j]
                    valid = valid & (dt <= W) & (dt >= 0.0)

                    u_j = u[j]
                    e_j = e[j]

                    g_val = g_tilde_scalar(dt)
                    z_val = z_scalar(dt)

                    colvec = G_node[:, u_j]                            # (N,)
                    outer_em = (colvec[:, None]) * (M_K[e_j, :][None, :])  # (N,M)

                    excite_mat_new = vals[0] + (alpha * g_val) * outer_em
                    z_accum_new    = vals[1] + colvec * (z_val * signs[e_j])

                    excite_mat = jnp.where(valid, excite_mat_new, vals[0])
                    z_accum    = jnp.where(valid, z_accum_new, vals[1])
                    return (excite_mat, z_accum), None

                (excite_mat, z_accum), _ = lax.scan(body_j, init=(excite_mat, z_accum), xs=jnp.arange(Lg_max))

                eta = mu + excite_mat + gamma * (z_accum ** 2)[:, None]  # (N,M)
                lam = phi_link(eta)
                lam_sum = jnp.sum(lam)
                return carry, lam_sum

            _, lam_series = lax.scan(step_grid, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(grid_times.shape[0]))
            dt = jnp.diff(grid_times)
            avg = 0.5 * (lam_series[:-1] + lam_series[1:])
            return jnp.sum(avg * dt)

        comp_int = compensator_grid()
        loglik = event_loglik - comp_int

    numpyro.factor("loglik", loglik)


# ---------------- Main driver ----------------
def main():
    p = argparse.ArgumentParser(description="Nonparametric kernels Hawkes with finite look-back window (static-bounds), normalized kernels + global amplitude.")
    p.add_argument("--data", type=str, default="traffic_hawkes_simulation2.pickle")
    p.add_argument("--method", type=str, choices=["mcmc", "map"], default="mcmc")
    p.add_argument("--warmup", type=int, default=3000)
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # basis choices
    p.add_argument("--B_t", type=int, default=20, help="# temporal Gaussian basis")
    p.add_argument("--B_s", type=int, default=20, help="# spatial Gaussian basis")
    p.add_argument("--time_scale", type=float, default=None, help="temporal basis width; default auto from T")
    p.add_argument("--space_scale", type=float, default=None, help="spatial basis width; default auto from max dist")
    # window (time units)
    p.add_argument("--window", type=float, default=10.0, help="finite look-back window W; default None = full history")
    # spatial normalization
    p.add_argument("--spatial_norm", type=str, choices=["mean", "sum"], default="sum",
                   help="Per-column normalization for κ̃ over reachable targets: mean or sum.")
    # QHP options (SOE form)
    p.add_argument("--use_qhp", action="store_true", help="Enable Quadratic Hawkes Z^2 term (sum of exponentials kernel).")
    p.add_argument("--B_q", type=int, default=0, help="# exponentials R for quadratic kernel z(τ)=Σ w_r β_r e^{-β_r τ}")
    p.add_argument("--q_scale", type=float, default=None, help="rate prior center ~ 1/q_scale for β_r")
    # Nonlinearity + numeric compensator grid
    p.add_argument("--nonlinearity", type=str,
                   choices=["linear", "softplus", "relu", "exp", "power2"],
                   default="linear", help="Link φ applied to intensity.")
    p.add_argument("--comp_grid", type=int, default=256,
                   help="Grid size for nonlinear compensator; will be unioned with event times.")
    # optional SVI warm start before MCMC
    p.add_argument("--svi_iters", type=int, default=0, help="Run SVI for this many iterations before MCMC; 0=skip")
    p.add_argument("--svi_lr", type=float, default=5e-2, help="SVI learning rate for warmup")
    # network connectivity
    p.add_argument("--num-hops", type=int, default=None, help="Override num_hops for reachability (default: read from data)")
    args = p.parse_args()

    enable_x64()

    # --- Load generator pickle
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    events = data["events"]
    num_nodes = int(data["num_nodes"])
    num_event_types = int(data["num_event_types"])
    node_locations = np.asarray(data["node_locations"], dtype=float)  # (N,2)
    adjacency = np.asarray(data["adjacency_matrix"], dtype=float)     # (N,N)
    # Override num_hops if specified
    num_hops = args.num_hops if args.num_hops is not None else int(data.get("num_hops", 1))
    print(f"Using num_hops: {num_hops}")
    params_init = data.get("params", None)

    # --- Prepare arrays
    t_np, u_np, e_np, T_np, N_from_ev, M_from_ev = prep_events_structured(events, num_event_types)
    assert num_nodes == N_from_ev
    assert num_event_types == M_from_ev

    # sort events by time (ensure monotone)
    order = np.argsort(t_np)
    t_np = t_np[order]; u_np = u_np[order]; e_np = e_np[order]

    reach_mask_np = compute_reachability(adjacency, num_hops=num_hops)

    # --- Window W and start indices (HOST / NumPy)
    if args.window is None:
        W = np.inf
    else:
        W = float(args.window)
    if np.isfinite(W):
        starts = np.searchsorted(t_np, t_np - W, side="left")
        starts = np.minimum(starts, np.arange(t_np.shape[0]))  # ensure j ≤ i
    else:
        starts = np.zeros_like(t_np, dtype=np.int64)
    L_max = int(np.max(np.arange(t_np.shape[0]) - starts)) if t_np.size else 0

    # --- Compensator grid (used only for non-linear links)
    nonlin = args.nonlinearity
    if nonlin != "linear":
        G = max(int(args.comp_grid), 2)
        uniform = np.linspace(0.0, T_np, G)
        grid_times_np = np.unique(np.concatenate([uniform, t_np]))
        if np.isfinite(W):
            grid_starts_np = np.searchsorted(t_np, grid_times_np - W, side="left")
        else:
            grid_starts_np = np.zeros_like(grid_times_np, dtype=np.int64)
        grid_ends_np = np.searchsorted(t_np, grid_times_np, side="left") - 1
        valid_len = np.maximum(grid_ends_np - grid_starts_np + 1, 0)
        Lg_max = int(valid_len.max()) if valid_len.size else 0
    else:
        grid_times_np = np.array([], dtype=np.float64)
        grid_starts_np = np.array([], dtype=np.int64)
        grid_ends_np = np.array([], dtype=np.int64)
        Lg_max = 0

    # --- JAX arrays
    key = jax.random.PRNGKey(args.seed)
    t = jnp.asarray(t_np)
    u = jnp.asarray(u_np)
    e = jnp.asarray(e_np)
    T = jnp.asarray(T_np, dtype=t.dtype)
    node_xy = jnp.asarray(node_locations)
    reach_mask = jnp.asarray(reach_mask_np)

    start_idx = jnp.asarray(starts, dtype=jnp.int32)
    L_max = int(L_max)
    W_jax = jnp.asarray(W, dtype=t.dtype)

    grid_times = jnp.asarray(grid_times_np)
    grid_starts = jnp.asarray(grid_starts_np, dtype=jnp.int32)
    grid_ends = jnp.asarray(grid_ends_np, dtype=jnp.int32)
    Lg_max = int(Lg_max)

    N = int(num_nodes)
    M = int(num_event_types)

    # --- Build basis
    # time centers in [0, T]
    B_t = int(args.B_t)
    time_centers = make_centers(T, B_t)
    if args.time_scale is None:
        time_scale = (T / max(B_t - 1, 1)) * 1.25
    else:
        time_scale = float(args.time_scale)
    time_scale = jnp.asarray(time_scale, dtype=t.dtype)

    # spatial centers in [0, r_max]
    D_np = np.sqrt(np.maximum(((node_locations[:, None, :] - node_locations[None, :, :]) ** 2).sum(-1), 0.0))
    r_max = float(D_np.max()) if D_np.size > 0 else 1.0
    B_s = int(args.B_s)
    space_centers = make_centers(r_max, B_s)
    if args.space_scale is None:
        space_scale = (r_max / max(B_s - 1, 1)) * 1.25
    else:
        space_scale = float(args.space_scale)
    space_scale = jnp.asarray(space_scale, dtype=t.dtype)

    # quadratic temporal "basis" (only its length R matters for SOE)
    if args.use_qhp and int(args.B_q) > 0:
        B_q = int(args.B_q)
        time_centers_q = make_centers(T, B_q)  # values unused; length sets R
        if args.q_scale is None:
            q_scale = (T / max(B_q - 1, 1)) * 1.25
        else:
            q_scale = float(args.q_scale)
        q_scale = jnp.asarray(q_scale, dtype=t.dtype)
    else:
        time_centers_q = jnp.asarray([], dtype=t.dtype)
        q_scale = jnp.asarray(1.0, dtype=t.dtype)

    # truths (if present; note σ/ω are not used directly in likelihood)
    mu_true, K_true, omega_true, sigma_true = reconstruct_true_params_if_present(
        np.asarray(params_init) if params_init is not None else None, N, M
    )
    M_K_true = np.asarray(data["mark_kernel_matrix"]) if "mark_kernel_matrix" in data else None

    # --- Inference
    if args.method == "mcmc":
        init_strategy = None
        if getattr(args, "svi_iters", 0) and args.svi_iters > 0:
            guide_warm = autoguide.AutoDelta(hawkes_np_model)
            svi_warm = SVI(hawkes_np_model, guide_warm, numpyro.optim.Adam(getattr(args, "svi_lr", 5e-2)), loss=Trace_ELBO())
            state = svi_warm.init(
                jax.random.PRNGKey(args.seed),
                t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                space_centers=space_centers, space_scale=space_scale,
                time_centers_q=time_centers_q, q_scale=q_scale,
                start_idx=start_idx, L_max=L_max, W=W_jax,
                N=N, M=M, spatial_norm=args.spatial_norm,
                nonlinearity=args.nonlinearity,
                grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max
            )
            for _ in range(args.svi_iters):
                state, _ = svi_warm.update(
                    state,
                    t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                    time_centers=time_centers, time_scale=time_scale,
                    space_centers=space_centers, space_scale=space_scale,
                    time_centers_q=time_centers_q, q_scale=q_scale,
                    start_idx=start_idx, L_max=L_max, W=W_jax,
                    N=N, M=M, spatial_norm=args.spatial_norm,
                    nonlinearity=args.nonlinearity,
                    grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max
                )
            init_params = guide_warm.median(svi_warm.get_params(state))
            init_strategy = init_to_value(values=init_params)
            print(f"Finished SVI warmup: {args.svi_iters} iters. Starting MCMC...")

        kernel = NUTS(hawkes_np_model, init_strategy=init_strategy, adapt_step_size=True, adapt_mass=True, regularize_mass_matrix=True) if init_strategy else NUTS(hawkes_np_model, target_accept_prob=0.85)
        mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                    num_chains=args.chains, chain_method="vectorized")
        mcmc.run(
            key,
            t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            space_centers=space_centers, space_scale=space_scale,
            time_centers_q=time_centers_q, q_scale=q_scale,
            start_idx=start_idx, L_max=L_max, W=W_jax,
            N=N, M=M, spatial_norm=args.spatial_norm,
            nonlinearity=args.nonlinearity,
            grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()

        mu_hat  = jnp.mean(posterior["mu"], axis=0)
        K_hat   = jnp.mean(posterior["K_masked"], axis=0)
        M_K_hat = jnp.mean(posterior["M_K"], axis=0)
        a_draws = posterior["a_uncon"]
        b_draws = posterior["b_uncon"]
        alpha_draws = posterior["alpha"]
        wq_draws = posterior.get("wq_uncon", None)
        betaq_draws = posterior.get("beta_q_det", None)
        gamma_draws = posterior.get("gamma", None)

        # Reconstruct posterior-mean κ̃ from b_uncon draws (to keep exactly same normalization rule)
        v_pos_hat = jnp.mean(jax.nn.softplus(b_draws) + 1e-8, axis=0)  # (B_s,)
        D = pairwise_dists(node_xy)
        psi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)  # (N,N,B_s)
        raw_kappa_hat = jnp.tensordot(psi, v_pos_hat, axes=[-1, 0])  # (N,N)
        mask = reach_mask
        col_counts = jnp.maximum(jnp.sum(mask, axis=0), 1.0)
        if args.spatial_norm == "sum":
            Z_s_col = jnp.maximum(jnp.sum(raw_kappa_hat * mask, axis=0), 1e-12)
        else:
            Z_s_col = jnp.sum(raw_kappa_hat * mask, axis=0) / col_counts
            Z_s_col = jnp.where(jnp.isfinite(Z_s_col), Z_s_col, 1.0)
        kappa_tilde_hat = raw_kappa_hat / Z_s_col[None, :]

        alpha_hat = float(jnp.mean(alpha_draws))
        gamma_hat = float(jnp.mean(gamma_draws)) if gamma_draws is not None else 0.0

        np.savez(
            f"mcmc_state_np_{args.data.split('.')[0]}.npz",
            mu=np.asarray(posterior["mu"]),
            K_masked=np.asarray(posterior["K_masked"]),
            M_K=np.asarray(posterior["M_K"]),
            a_uncon=np.asarray(a_draws),
            b_uncon=np.asarray(b_draws),
            alpha=np.asarray(alpha_draws),
            wq_uncon=np.asarray(wq_draws) if wq_draws is not None else None,
            beta_q=np.asarray(betaq_draws) if betaq_draws is not None else None,
            gamma=np.asarray(gamma_draws) if gamma_draws is not None else None,
            time_centers=np.asarray(time_centers),
            time_scale=float(time_scale),
            space_centers=np.asarray(space_centers),
            space_scale=float(space_scale),
            time_centers_q=np.asarray(time_centers_q),
            q_scale=float(q_scale),
            t=np.asarray(t), u=np.asarray(u), e=np.asarray(e), T=float(T),
            node_locations=np.asarray(node_locations),
            reach_mask=np.asarray(reach_mask_np),
            start_idx=np.asarray(starts),
            L_max=L_max,
            window=W if np.isfinite(W) else np.inf,
            mu_true=np.asarray(mu_true) if mu_true is not None else None,
            K_true=np.asarray(K_true) if K_true is not None else None,
            sigma_true=float(sigma_true) if sigma_true is not None else None,
            omega_true=float(omega_true) if omega_true is not None else None,
            M_K_true=np.asarray(M_K_true) if M_K_true is not None else None
        )
        print("Saved full MCMC posterior to mcmc_state_np_*.npz")

    else:
        guide = autoguide.AutoDelta(hawkes_np_model)
        svi = SVI(hawkes_np_model, guide, numpyro.optim.Adam(args.svi_lr), loss=Trace_ELBO())
        state = svi.init(
            jax.random.PRNGKey(args.seed),
            t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            space_centers=space_centers, space_scale=space_scale,
            time_centers_q=time_centers_q, q_scale=q_scale,
            start_idx=start_idx, L_max=L_max, W=W_jax,
            N=N, M=M, spatial_norm=args.spatial_norm,
            nonlinearity=args.nonlinearity,
            grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max
        )
        svi_iters = int(args.svi_iters) if getattr(args, "svi_iters", 0) and args.svi_iters > 0 else 2000
        for i in range(svi_iters):
            state, loss = svi.update(
                state,
                t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                space_centers=space_centers, space_scale=space_scale,
                time_centers_q=time_centers_q, q_scale=q_scale,
                start_idx=start_idx, L_max=L_max, W=W_jax,
                N=N, M=M, spatial_norm=args.spatial_norm,
                nonlinearity=args.nonlinearity,
                grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max
            )
            if (i + 1) % 200 == 0:
                print(f"[SVI] iter {i+1:04d} loss={float(loss):.3f}")
        params_map = svi.get_params(state)
        mu_hat  = params_map["mu"]
        K_hat   = params_map["K_masked"]
        M_K_hat = params_map["M_K"]
        # reconstruct α and κ̃ from MAP params
        alpha_hat = float(params_map["alpha"]) if "alpha" in params_map else 0.0
        v_pos_hat = jax.nn.softplus(params_map["b_uncon"]) + 1e-8
        D = pairwise_dists(node_xy)
        psi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)
        raw_kappa_hat = jnp.tensordot(psi, v_pos_hat, axes=[-1, 0])
        mask = reach_mask
        col_counts = jnp.maximum(jnp.sum(mask, axis=0), 1.0)
        if args.spatial_norm == "sum":
            Z_s_col = jnp.maximum(jnp.sum(raw_kappa_hat * mask, axis=0), 1e-12)
        else:
            Z_s_col = jnp.sum(raw_kappa_hat * mask, axis=0) / col_counts
            Z_s_col = jnp.where(jnp.isfinite(Z_s_col), Z_s_col, 1.0)
        kappa_tilde_hat = raw_kappa_hat / Z_s_col[None, :]
        gamma_hat = float(params_map["gamma"]) if ("gamma" in params_map) else 0.0

    # --- Report & save posterior means
    print("\n=== Posterior means (nonparam kernels, windowed, normalized) ===")
    print(f"mu_hat shape:  {tuple(np.asarray(mu_hat).shape)}")
    print(f"K_hat shape:   {tuple(np.asarray(K_hat).shape)}")
    print(f"M_K_hat shape: {tuple(np.asarray(M_K_hat).shape)}")
    print(f"alpha_hat:     {alpha_hat:.6f}")
    print(f"spatial_norm:  {args.spatial_norm}")
    print(f"nonlinearity:  {args.nonlinearity}")

    # Create a clean filename that includes quadratic info
    model_name = args.nonlinearity
    if args.use_qhp and int(getattr(args, "B_q", 0)) > 0:
        model_name = f"{args.nonlinearity}_quad"

    out = {
        "mu_hat":    np.asarray(mu_hat),
        "K_hat":     np.asarray(K_hat),
        "M_K_hat":   np.asarray(M_K_hat),
        "alpha_hat": float(alpha_hat),
        "kappa_tilde_hat": np.asarray(kappa_tilde_hat),
        "N": N,
        "M": M,
        "T": float(T),
        "node_locations": np.asarray(node_locations),
        "reach_mask": np.asarray(reach_mask_np),
        "data_pickle": args.data,
        "method": args.method,
        "time_centers": np.asarray(time_centers),
        "time_scale": float(time_scale),
        "space_centers": np.asarray(space_centers),
        "space_scale": float(space_scale),
        "window": W if np.isfinite(W) else np.inf,
        "L_max": L_max,
        "spatial_norm": args.spatial_norm,
        "nonlinearity": args.nonlinearity,
        "comp_grid": int(args.comp_grid),
    }
    if int(getattr(args, "B_q", 0)) > 0 and args.use_qhp:
        out.update({
            "gamma_hat": float(gamma_hat),
            "time_centers_q": np.asarray(time_centers_q),  # length R
            "q_scale": float(q_scale),
        })
    if "mcmc" in args.method:
        out["mcmc_state_file"] = f"mcmc_state_np_{args.data.split('.')[0]}_{model_name}.npz"
    
    with open(f"inference_result_np_{args.data.split('.')[0]}_{model_name}.pickle", "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved posterior means to inference_result_np_{args.data.split('.')[0]}_{model_name}.pickle")

    # ---------- Scale-aware comparisons ----------
    def rmse(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    # Only mu/K/M_K are directly comparable in structure; for K we provide scale-aware diagnostics.
    mu_true_cmp, K_true_cmp = None, None
    if params_init is not None:
        mu_true_cmp, K_true_cmp, omega_true_cmp, sigma_true_cmp = reconstruct_true_params_if_present(np.asarray(params_init), N, M)
    else:
        omega_true_cmp = sigma_true_cmp = None

    if mu_true_cmp is not None:
        rmse_mu = rmse(out["mu_hat"], mu_true_cmp)
        print(f"mu    RMSE vs true: {rmse_mu:.6f}")

    # --- K / G comparisons with EVAL FIX ---
    if (K_true_cmp is not None) and (sigma_true_cmp is not None):
        # Build true Gaussian spatial kernel κ_σ on the same node grid
        d2 = pairwise_sq_dists(jnp.asarray(node_locations))
        sigma_true_jax = jnp.asarray(sigma_true_cmp)
        denom = 2.0 * (sigma_true_jax ** 2)
        norm = 1.0 / (2.0 * jnp.pi * (sigma_true_jax ** 2))
        kappa_true = np.asarray(norm * jnp.exp(-d2 / denom))  # (N,N)

        mask = np.asarray(reach_mask_np)
        col_counts = np.maximum(mask.sum(axis=0), 1.0)

        # Normalize the TRUE spatial kernel using the SAME rule used in the model
        if args.spatial_norm == "sum":
            Z_true_col = (kappa_true * mask).sum(axis=0)
            Z_true_col = np.maximum(Z_true_col, 1e-12)
        else:  # "mean"
            Z_true_col = (kappa_true * mask).sum(axis=0) / col_counts
            Z_true_col = np.where(np.isfinite(Z_true_col), Z_true_col, 1.0)

        kappa_true_norm = kappa_true / Z_true_col[None, :]  # normalized like kappa_tilde_hat

        # Effective kernels
        G_eff_hat  = np.asarray(alpha_hat) * np.asarray(K_hat) * np.asarray(kappa_tilde_hat)     # (N,N)
        G_eff_true = np.asarray(K_true_cmp) * np.asarray(kappa_true_norm)                        # (N,N)

        # Raw RMSE (same normalization rule on both sides)
        rmse_Geff = rmse(G_eff_hat, G_eff_true)
        print(f"G_effective RMSE (α·K∘κ̃  vs  K_true∘κ_true^norm): {rmse_Geff:.6f}")

        # Best scalar rescale (to separate scale from shape)
        A = G_eff_hat
        B = G_eff_true
        c = float(np.sum(A * B) / max(np.sum(A * A), 1e-12))
        rmse_Geff_rescaled = rmse(c * A, B)
        print(f"best scalar c for G_effective: {c:.6f}")
        print(f"G_effective RMSE after best rescale: {rmse_Geff_rescaled:.6f}")

        # "Structural K" diagnostic
        K_struct_true = np.asarray(K_true_cmp) * Z_true_col[None, :]
        K_struct_hat  = float(alpha_hat) * np.asarray(K_hat)
        rmse_K_struct = rmse(K_struct_hat, K_struct_true)
        print(f"K structural RMSE (α·K  vs  K_true·Z_true_col): {rmse_K_struct:.6f}")

    # --- M_K comparison (note: K and M_K are still multiplicatively confounded) ---
    if "mark_kernel_matrix" in data:
        M_K_true = np.asarray(data["mark_kernel_matrix"])
        rmse_MK = rmse(out["M_K_hat"], M_K_true)
        print(f"M_K   RMSE vs true (scale may be confounded with K): {rmse_MK:.6f}")


if __name__ == "__main__":
    main()
