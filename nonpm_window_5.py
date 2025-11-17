#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint spatio-temporal Hawkes with an RKHS / GP prior on the kernel ψ̃(τ, r).

- ψ̃(τ, r) is represented on a 2D grid of temporal and distance centres:
    φ_t(τ; c_t, s_t) × φ_r(r; c_r, s_r)
- Coefficients W[k, m] over this grid are *not* i.i.d.:
    vec(W) ~ N(0, Σ_t ⊗ Σ_r)
  where Σ_t and Σ_r are SE kernels over time and distance centres.
- Identifiability: per-pair unit time integral
    ∫_0^∞ ψ̃(τ, r_ij) dτ = 1
  and a global amplitude α controls excitation scale.

Finite window W, reachability mask, and mark coupling as in the original code.
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

# ---------------- Platform ----------------
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(10)

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


def pairwise_dists(node_xy):
    diff = node_xy[:, None, :] - node_xy[None, :, :]
    d2 = jnp.sum(diff * diff, axis=-1)
    return jnp.sqrt(jnp.maximum(d2, 0.0))


# --------------- Gaussian basis utils ---------------
def make_centers(width, n):
    if n == 1:
        return jnp.array([0.5 * width])
    return jnp.linspace(0.0, width, n)


def gauss_bump(x, c, scale):
    z = (x - c) / scale
    return jnp.exp(-0.5 * z * z)


def gauss_bump_int_0_to(x, c, scale):
    rt2 = jnp.sqrt(2.0)
    pref = scale * jnp.sqrt(jnp.pi / 2.0)
    return pref * (erf((x - c) / (rt2 * scale)) - erf((-c) / (rt2 * scale)))


def gauss_bump_int_0_to_inf(c, scale):
    rt2 = jnp.sqrt(2.0)
    return scale * jnp.sqrt(jnp.pi / 2.0) * (1.0 - erf((-c) / (rt2 * scale)))


# --------------- Joint spatio-temporal kernel model (RKHS prior) ---------------
def hawkes_np_st_rkhs_model(
    t, u, e, T,
    node_xy, reach_mask,
    time_centers, time_scale,
    dist_centers, dist_scale,
    cov_W,
    start_idx, L_max, W,
    N: int, M: int,
):
    """
    Joint kernel ψ̃(τ, r) with per-pair unit time integral and RKHS/GP prior on W.

        λ_{i,e}(t) = μ_{i,e}
                     + α * Σ_{j: t_j<t} K_{i,u_j} * M_K[e_j,e] * ψ̃(t - t_j, r_{i,u_j})

    Windowed: only terms with (t - t_j) <= W are included.
    """
    Kevents = t.shape[0]

    # ---- Baselines and couplings ----
    mu_uncon = numpyro.sample("mu_uncon", dist.Normal(0.0, 1.0).expand([N, M]).to_event(2))
    mu = numpyro.deterministic("mu", jax.nn.softplus(mu_uncon) + 1e-8)

    K_uncon = numpyro.sample("K_uncon", dist.Normal(0.0, 1.0).expand([N, N]).to_event(2))
    K_pos = jax.nn.softplus(K_uncon)
    K_pre = K_pos * reach_mask
    colsum_K = jnp.maximum(jnp.sum(K_pre, axis=0), 1e-12)
    K_masked = numpyro.deterministic("K_masked", K_pre / colsum_K[None, :])

    M_uncon = numpyro.sample("M_uncon", dist.Normal(0.0, 1.0).expand([M, M]).to_event(2))
    M_pos = jax.nn.softplus(M_uncon) + 1e-8
    rowsum_M = jnp.maximum(jnp.sum(M_pos, axis=1), 1e-12)
    M_K = numpyro.deterministic("M_K", M_pos / rowsum_M[:, None])

    alpha = numpyro.sample("alpha", dist.Beta(2.0, 4.0))

    # ---- RKHS / GP prior for joint kernel coefficients W ----
    B_t = time_centers.shape[0]
    B_r = dist_centers.shape[0]
    Kdim = B_t * B_r

    # vec(W_uncon) ~ N(0, cov_W)
    zero_mean = jnp.zeros(Kdim)
    W_flat = numpyro.sample(
        "W_uncon",
        dist.MultivariateNormal(loc=zero_mean, covariance_matrix=cov_W)
    )
    W_uncon = W_flat.reshape((B_t, B_r))
    w_pos = jax.nn.softplus(W_uncon) + 1e-8  # enforce nonnegativity

    # ---- Precompute spatial basis per pair (N,N,B_r) ----
    D = pairwise_dists(node_xy)
    Psi_r = jnp.stack([gauss_bump(D, c, dist_scale) for c in dist_centers], axis=-1)  # (N,N,B_r)

    # For each pair (i,j), collect coefficients along time basis:
    # S_t[i,j,:] = Σ_b w_pos[:, b] * Psi_r[i,j,b]  -> shape (N,N,B_t)
    S_t = jnp.tensordot(Psi_r, w_pos, axes=[[2], [1]])  # (N,N,B_t)

    # ---- Per-pair normalisation: unit time integral ----
    I_inf = jnp.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers])  # (B_t,)
    denom = jnp.maximum(jnp.tensordot(S_t, I_inf, axes=[[2], [0]]), 1e-12)  # (N,N)
    numpyro.deterministic("kernel_denom_pair", denom)

    def phi_t(dt):  # (B_t,)
        dt = jnp.maximum(dt, 0.0)
        return jnp.exp(-0.5 * ((dt - time_centers) / time_scale) ** 2)

    def psi_val(dt, i_idx, j_idx):
        num = jnp.dot(S_t[i_idx, j_idx], phi_t(dt))
        return num / denom[i_idx, j_idx]

    def psi_int(dt_cap, i_idx, j_idx):  # ∫_0^{dt_cap} ψ̃(τ, r_ij) dτ
        dt_cap = jnp.maximum(dt_cap, 0.0)
        I_cap = jnp.stack([gauss_bump_int_0_to(dt_cap, c, time_scale) for c in time_centers], axis=0)
        num = jnp.dot(S_t[i_idx, j_idx], I_cap)
        return num / denom[i_idx, j_idx]

    # ---- Event log-likelihood with fixed-length scan per event ----
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
            val = psi_val(dt, u_i, u[j_clamped])
            contrib = K_masked[u_i, u[j_clamped]] * M_K[e[j_clamped], e_i] * (alpha * val)
            contrib = jnp.where(valid, contrib, jnp.array(0.0, dtype=t.dtype))
            return acc + contrib, None

        excite_sum, _ = lax.scan(body, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(L_max))
        lam_ie = mu[u_i, e_i] + excite_sum
        lam_ie = jnp.clip(lam_ie, a_min=1e-12)
        return carry + jnp.log(lam_ie), None

    event_loglik, _ = lax.scan(step_event, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))

    # ---- Compensator with window and pairwise dependence ----
    base_int = T * jnp.sum(mu)
    rowsum_MK = jnp.sum(M_K, axis=1)  # (M,)

    tail_limit = jnp.minimum(T - t, W)  # (K,)

    def comp_step(carry, j):
        u_j = u[j]
        e_j = e[j]
        cap = tail_limit[j]
        I_cap = jnp.stack([gauss_bump_int_0_to(cap, c, time_scale) for c in time_centers], axis=0)
        num_vec = jnp.dot(S_t[:, u_j, :], I_cap)  # (N,)
        J_vec = num_vec / denom[:, u_j]
        col = K_masked[:, u_j]
        col_sum = jnp.dot(col, J_vec)
        return carry + alpha * rowsum_MK[e_j] * col_sum, None

    exc_int, _ = lax.scan(comp_step, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))

    loglik = event_loglik - base_int - exc_int
    numpyro.factor("loglik", loglik)


# --------------- Joint kernel analysis utilities (unchanged) ---------------
def reconstruct_joint_kernel(W_uncon, time_centers, time_scale, dist_centers, dist_scale, 
                             tau_grid, r_grid):
    B_t, B_r = W_uncon.shape
    N_r, N_tau = len(r_grid), len(tau_grid)

    W_uncon = np.asarray(W_uncon)
    time_centers = np.asarray(time_centers)
    dist_centers = np.asarray(dist_centers)

    joint_kernel = np.zeros((N_r, N_tau))
    for i, r in enumerate(r_grid):
        for j, tau in enumerate(tau_grid):
            phi_r = np.exp(-0.5 * ((r - dist_centers) / dist_scale) ** 2)
            phi_tau = np.exp(-0.5 * ((tau - time_centers) / time_scale) ** 2)
            joint_kernel[i, j] = np.sum(W_uncon * np.outer(phi_tau, phi_r))
    return joint_kernel


def compute_kernel_statistics(W_uncon, time_centers, time_scale, dist_centers, dist_scale):
    W_uncon = np.asarray(W_uncon)
    time_centers = np.asarray(time_centers)
    dist_centers = np.asarray(dist_centers)

    weight_stats = {
        'mean': float(np.mean(W_uncon)),
        'std': float(np.std(W_uncon)),
        'min': float(np.min(W_uncon)),
        'max': float(np.max(W_uncon)),
        'sparsity': float(np.mean(W_uncon < 1e-6))
    }

    time_stats = {
        'centers': time_centers.tolist(),
        'scale': float(time_scale),
        'range': [float(time_centers.min()), float(time_centers.max())]
    }

    dist_stats = {
        'centers': dist_centers.tolist(),
        'scale': float(dist_scale),
        'range': [float(dist_centers.min()), float(dist_centers.max())]
    }

    interaction_stats = {
        'temporal_dominance': np.mean(np.abs(W_uncon), axis=1).tolist(),
        'spatial_dominance': np.mean(np.abs(W_uncon), axis=0).tolist(),
    }

    return {
        'weights': weight_stats,
        'temporal_basis': time_stats,
        'spatial_basis': dist_stats,
        'interactions': interaction_stats
    }


def analyze_kernel_behavior(W_uncon, time_centers, time_scale, dist_centers, dist_scale,
                            node_xy, tau_test=None, r_test=None):
    if tau_test is None:
        tau_test = np.linspace(0, 2 * float(time_centers.max()), 50)
    if r_test is None:
        r_test = np.linspace(0, 2 * float(dist_centers.max()), 50)

    joint_kernel = reconstruct_joint_kernel(W_uncon, time_centers, time_scale,
                                            dist_centers, dist_scale, tau_test, r_test)

    N = node_xy.shape[0]
    D_network = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D_network[i, j] = np.sqrt(np.sum((node_xy[i] - node_xy[j]) ** 2))

    dist_analysis = {
        'network_distances': D_network.flatten().tolist(),
        'mean_distance': float(np.mean(D_network)),
        'max_distance': float(np.max(D_network)),
        'distance_quantiles': np.percentile(D_network, [25, 50, 75]).tolist()
    }

    kernel_behavior = {}
    for r in [0.1, 0.5, 1.0, 2.0]:
        if r <= r_test.max():
            r_idx = np.argmin(np.abs(r_test - r))
            row = joint_kernel[r_idx, :]
            kernel_behavior[f'r_{r}'] = {
                'temporal_decay': row.tolist(),
                'peak_time': float(tau_test[np.argmax(row)]),
                'total_excitation': float(np.trapz(row, tau_test))
            }

    return {
        'distance_analysis': dist_analysis,
        'kernel_behavior': kernel_behavior,
        'test_grid': {
            'tau': tau_test.tolist(),
            'r': r_test.tolist()
        }
    }


# ---------------- Main driver ----------------
def main():
    p = argparse.ArgumentParser(description="Joint spatio-temporal Hawkes with RKHS/GP prior on kernel and finite window W.")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--method", type=str, choices=["mcmc", "map"], default="mcmc")
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    # basis
    p.add_argument("--B_t", type=int, default=16, help="# temporal basis")
    p.add_argument("--B_r", type=int, default=16, help="# distance basis")
    p.add_argument("--time_scale", type=float, default=None)
    p.add_argument("--dist_scale", type=float, default=None)
    # RKHS / GP hyperparams
    p.add_argument("--rkhs_sigma", type=float, default=1.0, help="Kernel amplitude for GP prior on W")
    p.add_argument("--rkhs_ell_t", type=float, default=None, help="Length scale in time for GP prior on W")
    p.add_argument("--rkhs_ell_r", type=float, default=None, help="Length scale in distance for GP prior on W")
    # window
    p.add_argument("--window", type=float, default=10.0)
    # optional SVI warm start
    p.add_argument("--svi_iters", type=int, default=0)
    p.add_argument("--svi_lr", type=float, default=5e-2)
    args = p.parse_args()

    enable_x64()

    # Load data
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    events = data["events"]
    num_nodes = int(data["num_nodes"])
    num_event_types = int(data["num_event_types"])
    node_locations = np.asarray(data["node_locations"], dtype=float)
    adjacency = np.asarray(data["adjacency_matrix"], dtype=float)
    num_hops = int(data.get("num_hops", 1))

    t_np, u_np, e_np, T_np, N_ev, M_ev = prep_events_structured(events, num_event_types)
    assert num_nodes == N_ev and num_event_types == M_ev

    order = np.argsort(t_np)
    t_np = t_np[order]; u_np = u_np[order]; e_np = e_np[order]

    reach_mask_np = compute_reachability(adjacency, num_hops=num_hops)

    # Window and start indices
    W = float(args.window) if args.window is not None else np.inf
    if np.isfinite(W):
        starts = np.searchsorted(t_np, t_np - W, side="left")
        starts = np.minimum(starts, np.arange(t_np.shape[0]))
    else:
        starts = np.zeros_like(t_np, dtype=np.int64)
    L_max = int(np.max(np.arange(t_np.shape[0]) - starts)) if t_np.size else 0

    # JAX arrays
    key = jax.random.PRNGKey(args.seed)
    t = jnp.asarray(t_np); u = jnp.asarray(u_np); e = jnp.asarray(e_np)
    T = jnp.asarray(T_np, dtype=t.dtype)
    node_xy = jnp.asarray(node_locations)
    reach_mask = jnp.asarray(reach_mask_np)

    start_idx = jnp.asarray(starts, dtype=jnp.int32)
    L_max = int(L_max)
    W_jax = jnp.asarray(W, dtype=t.dtype)

    N = int(num_nodes); M = int(num_event_types)

    # Basis construction
    B_t = int(args.B_t)
    time_centers = make_centers(T, B_t)
    if args.time_scale is None:
        time_scale = (T / max(B_t - 1, 1)) * 1.25
    else:
        time_scale = float(args.time_scale)
    time_scale = jnp.asarray(time_scale, dtype=t.dtype)

    D_np = np.sqrt(np.maximum(((node_locations[:, None, :] - node_locations[None, :, :]) ** 2).sum(-1), 0.0))
    r_max = float(D_np.max()) if D_np.size > 0 else 1.0
    B_r = int(args.B_r)
    dist_centers = make_centers(r_max, B_r)
    if args.dist_scale is None:
        dist_scale = (r_max / max(B_r - 1, 1)) * 1.25
    else:
        dist_scale = float(args.dist_scale)
    dist_scale = jnp.asarray(dist_scale, dtype=t.dtype)

    # ---- RKHS / GP covariance for vec(W) ----
    time_centers_np = np.array(time_centers, dtype=float)
    dist_centers_np = np.array(dist_centers, dtype=float)

    sigma_w = float(args.rkhs_sigma)
    ell_t = float(args.rkhs_ell_t) if args.rkhs_ell_t is not None else float(time_scale)
    ell_r = float(args.rkhs_ell_r) if args.rkhs_ell_r is not None else float(dist_scale)

    # SE kernels over time and distance centres
    diff_t = time_centers_np[:, None] - time_centers_np[None, :]
    cov_t = np.exp(-0.5 * (diff_t ** 2) / (ell_t ** 2 + 1e-12))

    diff_r = dist_centers_np[:, None] - dist_centers_np[None, :]
    cov_r = np.exp(-0.5 * (diff_r ** 2) / (ell_r ** 2 + 1e-12))

    # Kronecker covariance for vec(W) (time-major or distance-major is fine as long as reshape is consistent)
    cov_W_np = sigma_w ** 2 * np.kron(cov_r, cov_t)
    Kdim = B_t * B_r
    cov_W_np += 1e-6 * np.eye(Kdim)  # jitter for PD

    cov_W = jnp.asarray(cov_W_np, dtype=t.dtype)

    # Inference
    if args.method == "mcmc":
        init_strategy = None
        if args.svi_iters and args.svi_iters > 0:
            guide_warm = autoguide.AutoDelta(hawkes_np_st_rkhs_model)
            svi_warm = SVI(hawkes_np_st_rkhs_model, guide_warm, numpyro.optim.Adam(args.svi_lr), loss=Trace_ELBO())
            state = svi_warm.init(
                jax.random.PRNGKey(args.seed),
                t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                dist_centers=dist_centers, dist_scale=dist_scale,
                cov_W=cov_W,
                start_idx=start_idx, L_max=L_max, W=W_jax,
                N=N, M=M,
            )
            for _ in range(args.svi_iters):
                state, _ = svi_warm.update(
                    state,
                    t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                    time_centers=time_centers, time_scale=time_scale,
                    dist_centers=dist_centers, dist_scale=dist_scale,
                    cov_W=cov_W,
                    start_idx=start_idx, L_max=L_max, W=W_jax,
                    N=N, M=M,
                )
            init_params = guide_warm.median(svi_warm.get_params(state))
            init_strategy = init_to_value(values=init_params)
            print(f"Finished SVI warmup: {args.svi_iters} iters. Starting MCMC...")

        kernel = NUTS(hawkes_np_st_rkhs_model, target_accept_prob=0.85, init_strategy=init_strategy) \
            if init_strategy else NUTS(hawkes_np_st_rkhs_model, target_accept_prob=0.85)

        mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains, chain_method="parallel")
        mcmc.run(
            key,
            t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            dist_centers=dist_centers, dist_scale=dist_scale,
            cov_W=cov_W,
            start_idx=start_idx, L_max=L_max, W=W_jax,
            N=N, M=M,
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()

        mu_hat = jnp.mean(posterior["mu"], axis=0)
        K_hat = jnp.mean(posterior["K_masked"], axis=0)
        M_K_hat = jnp.mean(posterior["M_K"], axis=0)
        W_uncon_draws = posterior["W_uncon"]
        alpha_draws = posterior["alpha"]
        alpha_hat = float(jnp.mean(alpha_draws))
        W_uncon_hat = jnp.mean(W_uncon_draws, axis=0)

        np.savez(
            "mcmc_state_np3_rkhs.npz",
            mu=np.asarray(posterior["mu"]),
            K_masked=np.asarray(posterior["K_masked"]),
            M_K=np.asarray(posterior["M_K"]),
            W_uncon=np.asarray(W_uncon_draws),
            alpha=np.asarray(alpha_draws),
            time_centers=np.asarray(time_centers),
            time_scale=float(time_scale),
            dist_centers=np.asarray(dist_centers),
            dist_scale=float(dist_scale),
            t=np.asarray(t), u=np.asarray(u), e=np.asarray(e), T=float(T),
            node_locations=np.asarray(node_locations),
            reach_mask=np.asarray(reach_mask_np),
            start_idx=np.asarray(starts), L_max=L_max, window=W if np.isfinite(W) else np.inf,
        )
        print("Saved full MCMC posterior to mcmc_state_np3_rkhs.npz")

    else:
        guide = autoguide.AutoDelta(hawkes_np_st_rkhs_model)
        svi = SVI(hawkes_np_st_rkhs_model, guide, numpyro.optim.Adam(args.svi_lr), loss=Trace_ELBO())
        state = svi.init(
            jax.random.PRNGKey(args.seed),
            t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            dist_centers=dist_centers, dist_scale=dist_scale,
            cov_W=cov_W,
            start_idx=start_idx, L_max=L_max, W=W_jax,
            N=N, M=M,
        )
        svi_iters = int(args.svi_iters) if args.svi_iters and args.svi_iters > 0 else 2000
        for i in range(svi_iters):
            state, loss = svi.update(
                state,
                t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                dist_centers=dist_centers, dist_scale=dist_scale,
                cov_W=cov_W,
                start_idx=start_idx, L_max=L_max, W=W_jax,
                N=N, M=M,
            )
            if (i + 1) % 200 == 0:
                print(f"[SVI] iter {i+1:04d} loss={float(loss):.3f}")
        params_map = svi.get_params(state)
        mu_hat = params_map["mu"]
        K_hat = params_map["K_masked"]
        M_K_hat = params_map["M_K"]
        W_uncon_hat = params_map["W_uncon"]
        alpha_hat = float(params_map["alpha"])

    # Save posterior means + kernel analysis
    out = {
        "mu_hat": np.asarray(mu_hat),
        "K_hat": np.asarray(K_hat),
        "M_K_hat": np.asarray(M_K_hat),
        "alpha_hat": float(alpha_hat),
        "N": N, "M": M, "T": float(T),
        "node_locations": np.asarray(node_locations),
        "reach_mask": np.asarray(reach_mask_np),
        "data_pickle": args.data,
        "method": args.method,
        "time_centers": np.asarray(time_centers),
        "time_scale": float(time_scale),
        "dist_centers": np.asarray(dist_centers),
        "dist_scale": float(dist_scale),
        "window": W if np.isfinite(W) else np.inf,
        "L_max": L_max,
        "kernel_param": np.asarray(W_uncon_hat),  # (B_t,B_r)
        "mcmc_state_file": "mcmc_state_np3_rkhs.npz" if args.method == "mcmc" else None,
        "rkhs_sigma": sigma_w,
        "rkhs_ell_t": ell_t,
        "rkhs_ell_r": ell_r,
    }

    print("Computing enhanced joint kernel analysis...")
    try:
        kernel_stats = compute_kernel_statistics(W_uncon_hat, time_centers, time_scale,
                                                 dist_centers, dist_scale)
        out["kernel_statistics"] = kernel_stats

        kernel_behavior = analyze_kernel_behavior(W_uncon_hat, time_centers, time_scale,
                                                  dist_centers, dist_scale, node_locations)
        out["kernel_behavior"] = kernel_behavior

        print("  ✓ Kernel statistics computed")
        print("  ✓ Kernel behavior analyzed")

        print(f"\n=== JOINT KERNEL SUMMARY (RKHS prior) ===")
        print(f"Temporal bases: {len(time_centers)} (scale: {float(time_scale):.3g})")
        print(f"Distance bases: {len(dist_centers)} (scale: {float(dist_scale):.3g})")
        print(f"Weight statistics:")
        print(f"  Mean: {kernel_stats['weights']['mean']:.4f}")
        print(f"  Std:  {kernel_stats['weights']['std']:.4f}")
        print(f"  Sparsity: {kernel_stats['weights']['sparsity']:.1%}")
        print(f"Network distance analysis:")
        print(f"  Mean: {kernel_behavior['distance_analysis']['mean_distance']:.3f}")
        print(f"  Max:  {kernel_behavior['distance_analysis']['max_distance']:.3f}")
        print("=========================================\n")

    except Exception as e:
        print(f"  ⚠ Enhanced analysis failed: {e}")
        out["kernel_statistics"] = None
        out["kernel_behavior"] = None

    out_name = f"inference_result_np3_rkhs_{args.data.split('.')[0]}.pickle"
    with open(out_name, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved posterior means to {out_name}")


if __name__ == "__main__":
    main()
