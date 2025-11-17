#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marked multivariate Hawkes with nonparametric temporal & spatial kernels,
and a finite time window W so each event only depends on the last W seconds.

IDENTIFIABILITY-FIXED VERSION:
- Temporal kernel shape g̃ has unit integral (∫ g̃ = 1); spatial kernel κ̃ has mean 1
  over observed pairwise distances (per source column); a single scalar amplitude α
  carries total scale. This prevents κ/temporal from soaking up arbitrary amplitude.

- g̃(τ) = sum_k mix_w_k * N(τ; c_k, ℓ_t^2), τ >= 0, with ∫ g̃ = 1
- κ̃(r) = [sum_m v_m * N(r; s_m, ρ_s^2)] / mean_{reachable n}(...)  so col-mean(κ̃)=1
- Effective excitation uses α * g̃ and κ̃
- G_node = K_masked ∘ κ̃(distances)
- Windowed likelihood: only pairs with (t_i - t_j) <= W contribute.
  The compensator uses α * ∫_0^{min(T - t_j, W)} g̃(τ) dτ.

Fixes reverse-mode AD issue by removing dynamic loop bounds:
- Precompute start indices start_idx[i] on host with numpy.
- Inside the model, sum over a fixed K_max = max_i(i - start_idx[i]) with masking.

Saves:
- Posterior means to inference_result_np.pickle
- Full posterior draws to mcmc_state_np.npz

Reporting:
- Adds scale-aware RMSEs that compare EFFECTIVE kernels:
    RMSE_eff = RMSE( α_hat * K_hat ∘ κ̃_hat , K_true ∘ κ_σ_true )
  and an optional "structural" RMSE on K after rescaling K_true by
  the column means of κ_σ_true and dividing by α_hat.
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
    start_idx,  # (K,) int32/64 array, precomputed on host
    L_max,      # scalar int: max(i - start_idx[i])
    W,          # scalar (float) window; may be inf
    N: int, M: int
):
    """
    Nonparametric temporal g̃(τ) (unit integral) and spatial κ̃(r) (mean 1 per source column),
    with a finite window W. Effective kernel uses α * g̃ and κ̃. Uses fixed-length scan per
    event and masks terms with j < start_i.
    """
    Kevents = t.shape[0]

    # ---- Base rates and coupling matrices (MATCH parametric priors & transforms)
    mu_uncon = numpyro.sample("mu_uncon", dist.Normal(0.0, 1.0).expand([N, M]).to_event(2))
    mu = numpyro.deterministic("mu", jax.nn.softplus(mu_uncon) + 1e-6)  # (N,M)

    # EXACTLY like parametric: Normal(0, 0.5), softplus(+eps), then mask
    K_uncon = numpyro.sample("K_uncon", dist.Normal(0.0, 0.5).expand([N, N]).to_event(2))
    K_pos = jax.nn.softplus(K_uncon)
    K_masked = numpyro.deterministic("K_masked", K_pos * reach_mask)     # (N,N)

    # EXACTLY like parametric: Normal(0, 0.5), softplus(+eps)
    M_uncon = numpyro.sample("M_uncon", dist.Normal(0.0, 0.5).expand([M, M]).to_event(2))
    M_K = numpyro.deterministic("M_K", jax.nn.softplus(M_uncon) + 1e-6)  # (M,M)

    # ---- Global amplitude α (total excitation scale; encourages subcriticality)
    alpha = numpyro.sample("alpha", dist.Beta(2.0, 4.0))  # mean ~ 0.33

    # ---- Spatial kernel κ̃(r): per-source (column) mean = 1 over reachable pairs
    B_s = space_centers.shape[0]
    b_uncon = numpyro.sample("b_uncon", dist.Normal(0.0, 0.8).expand([B_s]).to_event(1))
    v_pos = jax.nn.softplus(b_uncon) + 1e-8

    D = pairwise_dists(node_xy)  # (N,N)
    psi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)  # (N,N,B_s)
    raw_kappa = jnp.tensordot(psi, v_pos, axes=[-1, 0])  # (N,N)

    # Normalize so for each source node j, mean_n κ̃[n,j] over reachable n is 1.
    mask = reach_mask  # (N,N) in {0,1}
    col_counts = jnp.sum(mask, axis=0)                          # (N,)
    col_counts = jnp.maximum(col_counts, 1.0)                   # avoid /0 for isolated columns
    Z_s_col = jnp.sum(raw_kappa * mask, axis=0) / col_counts    # (N,)
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
        lam_ie = mu[u_i, e_i] + excite_sum
        lam_ie = jnp.clip(lam_ie, a_min=1e-12)
        return carry + jnp.log(lam_ie), None

    event_loglik, _ = lax.scan(step_event, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))

    # ---- Compensator (integral) with window
    base_int = T * jnp.sum(mu)
    colsum_G = jnp.sum(G_node, axis=0)  # (N,)
    rowsum_MK = jnp.sum(M_K, axis=1)    # (M,)
    tail_limit = jnp.minimum(T - t, W)  # (K,)
    tail = alpha * G_tilde_int_vec(tail_limit)  # (K,)
    exc_int = jnp.sum(colsum_G[u] * rowsum_MK[e] * tail)

    loglik = event_loglik - base_int - exc_int
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
    num_hops = int(data.get("num_hops", 1))
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

    # truths (if present; note σ/ω are not used here)
    mu_true, K_true, omega_true, sigma_true = reconstruct_true_params_if_present(
        np.asarray(params_init) if params_init is not None else None, N, M
    )
    M_K_true = np.asarray(data["mark_kernel_matrix"]) if "mark_kernel_matrix" in data else None

    # --- Inference
    computed_tau_grid = None
    computed_g_tilde_grid = None

    if args.method == "mcmc":
        kernel = NUTS(hawkes_np_model, target_accept_prob=0.85)
        mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                    num_chains=args.chains, chain_method="parallel")
        mcmc.run(
            key,
            t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            space_centers=space_centers, space_scale=space_scale,
            start_idx=start_idx, L_max=L_max, W=W_jax,
            N=N, M=M
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()

        mu_hat  = jnp.mean(posterior["mu"], axis=0)
        K_hat   = jnp.mean(posterior["K_masked"], axis=0)
        M_K_hat = jnp.mean(posterior["M_K"], axis=0)
        a_draws = posterior["a_uncon"]
        b_draws = posterior["b_uncon"]
        alpha_draws = posterior["alpha"]

        # Reconstruct posterior-mean κ̃ from b_uncon draws
        v_pos_hat = jnp.mean(jax.nn.softplus(b_draws) + 1e-8, axis=0)  # (B_s,)
        D = pairwise_dists(node_xy)
        psi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)  # (N,N,B_s)
        raw_kappa_hat = jnp.tensordot(psi, v_pos_hat, axes=[-1, 0])  # (N,N)
        mask = reach_mask
        col_counts = jnp.maximum(jnp.sum(mask, axis=0), 1.0)
        Z_s_col = jnp.sum(raw_kappa_hat * mask, axis=0) / col_counts
        Z_s_col = jnp.where(jnp.isfinite(Z_s_col), Z_s_col, 1.0)
        kappa_tilde_hat = raw_kappa_hat / Z_s_col[None, :]  # (N,N)

        alpha_hat = float(jnp.mean(alpha_draws))

        # Temporal shape on a grid (posterior mean of weights)
        w_pos_mean = jnp.mean(jax.nn.softplus(a_draws) + 1e-8, axis=0)  # (B_t,)
        ints = jnp.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers])
        mix_w_mean = w_pos_mean / (jnp.dot(w_pos_mean, ints) + 1e-12)
        tau_grid = jnp.linspace(0.0, T, 200)
        phi = jnp.exp(-0.5 * ((tau_grid[:, None] - time_centers[None, :]) / time_scale) ** 2)
        g_tilde_grid = phi @ mix_w_mean
        computed_tau_grid = np.asarray(tau_grid)
        computed_g_tilde_grid = np.asarray(g_tilde_grid)

        np.savez(
            "mcmc_state_np.npz",
            mu=np.asarray(posterior["mu"]),
            K_masked=np.asarray(posterior["K_masked"]),
            M_K=np.asarray(posterior["M_K"]),
            a_uncon=np.asarray(a_draws),
            b_uncon=np.asarray(b_draws),
            alpha=np.asarray(alpha_draws),
            time_centers=np.asarray(time_centers),
            time_scale=float(time_scale),
            space_centers=np.asarray(space_centers),
            space_scale=float(space_scale),
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
        print("Saved full MCMC posterior to mcmc_state_np.npz")

    else:
        guide = autoguide.AutoDelta(hawkes_np_model)
        svi = SVI(hawkes_np_model, guide, numpyro.optim.Adam(5e-2), loss=Trace_ELBO())
        state = svi.init(
            jax.random.PRNGKey(args.seed),
            t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            space_centers=space_centers, space_scale=space_scale,
            start_idx=start_idx, L_max=L_max, W=W_jax,
            N=N, M=M
        )
        for i in range(2000):
            state, loss = svi.update(
                state,
                t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                space_centers=space_centers, space_scale=space_scale,
                start_idx=start_idx, L_max=L_max, W=W_jax,
                N=N, M=M
            )
            if (i + 1) % 200 == 0:
                print(f"[SVI] iter {i+1:04d} loss={float(loss):.3f}")
        params_map = svi.get_params(state)
        mu_hat  = params_map["mu"]
        K_hat   = params_map["K_masked"]
        M_K_hat = params_map["M_K"]
        # reconstruct α and κ̃ from MAP params
        alpha_hat = float(params_map["alpha"])
        # rebuild kappa_tilde_hat using MAP b_uncon
        v_pos_hat = jax.nn.softplus(params_map["b_uncon"]) + 1e-8
        D = pairwise_dists(node_xy)
        psi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)
        raw_kappa_hat = jnp.tensordot(psi, v_pos_hat, axes=[-1, 0])
        mask = reach_mask
        col_counts = jnp.maximum(jnp.sum(mask, axis=0), 1.0)
        Z_s_col = jnp.sum(raw_kappa_hat * mask, axis=0) / col_counts
        Z_s_col = jnp.where(jnp.isfinite(Z_s_col), Z_s_col, 1.0)
        kappa_tilde_hat = raw_kappa_hat / Z_s_col[None, :]

        # Temporal shape on a grid (MAP weights)
        w_pos_map = jax.nn.softplus(params_map["a_uncon"]) + 1e-8
        ints = jnp.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers])
        mix_w_map = w_pos_map / (jnp.dot(w_pos_map, ints) + 1e-12)
        tau_grid = jnp.linspace(0.0, T, 200)
        phi = jnp.exp(-0.5 * ((tau_grid[:, None] - time_centers[None, :]) / time_scale) ** 2)
        g_tilde_grid = phi @ mix_w_map
        computed_tau_grid = np.asarray(tau_grid)
        computed_g_tilde_grid = np.asarray(g_tilde_grid)

    # --- Report & save posterior means
    print("\n=== Posterior means (nonparam kernels, windowed, normalized) ===")
    print(f"mu_hat shape:  {tuple(np.asarray(mu_hat).shape)}")
    print(f"K_hat shape:   {tuple(np.asarray(K_hat).shape)}")
    print(f"M_K_hat shape: {tuple(np.asarray(M_K_hat).shape)}")
    print(f"alpha_hat:     {float(alpha_hat):.6f}")

    out = {
        "mu_hat":    np.asarray(mu_hat),
        "K_hat":     np.asarray(K_hat),
        "M_K_hat":   np.asarray(M_K_hat),
        "alpha_hat": float(alpha_hat),
        "kappa_tilde_hat": np.asarray(kappa_tilde_hat),
        "tau_grid":  np.asarray(computed_tau_grid) if computed_tau_grid is not None else None,
        "g_tilde_grid": np.asarray(computed_g_tilde_grid) if computed_g_tilde_grid is not None else None,
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
    }
    if "mcmc" in args.method:
        out["mcmc_state_file"] = "mcmc_state_np.npz"

    with open("inference_result_np.pickle", "wb") as f:
        pickle.dump(out, f)
    print("\nSaved posterior means to inference_result_np.pickle")

    # ---------- Scale-aware comparisons ----------
    def rmse(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    mu_true_cmp, K_true_cmp = None, None
    if params_init is not None:
        mu_true_cmp, K_true_cmp, omega_true_cmp, sigma_true_cmp = reconstruct_true_params_if_present(np.asarray(params_init), N, M)
    else:
        omega_true_cmp = sigma_true_cmp = None

    print("\n--- True vs Inferred (structure) ---")
    if mu_true_cmp is not None:
        rmse_mu = rmse(out["mu_hat"], mu_true_cmp)
        print(f"mu:    RMSE={rmse_mu:.6f}")

    # --- K comparison (scale-aware) ---
    if (K_true_cmp is not None) and (sigma_true_cmp is not None):
        # Build true Gaussian spatial kernel κ_σ on the same node grid
        d2 = pairwise_sq_dists(jnp.asarray(node_locations))
        sigma_true_jax = jnp.asarray(sigma_true_cmp)
        # normalized 2D Gaussian kernel as in the parametric script
        denom = 2.0 * (sigma_true_jax ** 2)
        norm = 1.0 / (2.0 * jnp.pi * (sigma_true_jax ** 2))
        kappa_true = np.asarray(norm * jnp.exp(-d2 / denom))  # (N,N)

        # Effective kernels
        G_eff_hat = np.asarray(alpha_hat) * np.asarray(K_hat) * np.asarray(kappa_tilde_hat)    # (N,N)
        G_eff_true = np.asarray(K_true_cmp) * np.asarray(kappa_true)                           # (N,N)

        rmse_Geff = rmse(G_eff_hat, G_eff_true)
        relfro_G = float(np.linalg.norm(G_eff_hat - G_eff_true) /
                         max(np.linalg.norm(G_eff_true), 1e-12))
        print(f"G_effective: RMSE={rmse_Geff:.6f}  rel.Fro.err={relfro_G:.3%}")

        # Optional: "structural K" diagnostic after aligning spatial col-mean and α
        mask = np.asarray(reach_mask_np)
        col_counts = np.maximum(mask.sum(axis=0), 1.0)
        kappa_col_means = (kappa_true * mask).sum(axis=0) / col_counts   # (N,)
        K_target = (np.asarray(K_true_cmp) * kappa_col_means[None, :]) / max(alpha_hat, 1e-12)
        rmse_K_struct = rmse(np.asarray(K_hat), K_target)
        relfro_K_struct = float(np.linalg.norm(np.asarray(K_hat) - K_target) /
                                max(np.linalg.norm(K_target), 1e-12))
        print(f"K structural: RMSE={rmse_K_struct:.6f}  rel.Fro.err={relfro_K_struct:.3%}")

    # --- M_K comparison (note: K and M_K are still multiplicatively confounded) ---
    if "mark_kernel_matrix" in data:
        M_K_true = np.asarray(data["mark_kernel_matrix"])
        rmse_MK = rmse(out["M_K_hat"], M_K_true)
        relfro_MK = float(np.linalg.norm(np.asarray(M_K_hat) - M_K_true) /
                          max(np.linalg.norm(M_K_true), 1e-12))
        print(f"M_K:         RMSE={rmse_MK:.6f}  rel.Fro.err={relfro_MK:.3%}")

    # Optional: dump arrays for quick inspection
    try:
        np.save("K_hat.npy", np.asarray(K_hat))
        np.save("M_K_hat.npy", np.asarray(M_K_hat))
        np.save("kappa_tilde_hat.npy", np.asarray(kappa_tilde_hat))
        if computed_g_tilde_grid is not None:
            np.save("tau_grid.npy", np.asarray(computed_tau_grid))
            np.save("g_tilde_grid.npy", np.asarray(computed_g_tilde_grid))
    except Exception as _:
        pass

if __name__ == "__main__":
    main()
