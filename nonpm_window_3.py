#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fully nonparametric spatio-temporal Hawkes with a joint kernel ψ̃(τ, r).
- Single kernel over lag τ >= 0 and pairwise distance r = ||x_i - x_j||.
- ψ̃ is parameterized with a nonnegative mixture over 2D Gaussian basis:
    φ_t(τ; c_t, s_t) × φ_r(r; c_r, s_r)
- Identifiability: unit time integral per pair (for each (i,j)): ∫_0^∞ ψ̃(τ, r_{ij}) dτ = 1
  A global amplitude α and pairwise coupling K carry overall excitation scale.
- Finite window W (seconds/hours): only pairs with (t_i - t_j) <= W contribute.
- Mark interactions via a nonnegative 2×2 matrix M_K.

Input format matches nonpm_window_2.py.
Saves posterior means to inference_result_np3_<data>.pickle and full state for MCMC.
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


# --------------- Joint spatio-temporal kernel model ---------------
def hawkes_np_st_model(
    t, u, e, T,
    node_xy, reach_mask,
    time_centers, time_scale,
    dist_centers, dist_scale,
    start_idx, L_max, W,
    N: int, M: int,
):
    """
    Joint kernel ψ̃(τ, r) with per-pair unit time integral. Excitation:
        λ_{i,e}(t) = μ_{i,e} + α * Σ_{j: t_j<t} K_{i,u_j} * M_K[e_j,e] * ψ̃(t - t_j, r_{i,u_j})
    Windowed: include terms only if (t - t_j) <= W.
    """
    Kevents = t.shape[0]

    # Base rates and couplings
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

    # Joint kernel weights (B_t × B_r)
    B_t = time_centers.shape[0]
    B_r = dist_centers.shape[0]
    W_uncon = numpyro.sample("W_uncon", dist.Normal(0.0, 0.8).expand([B_t, B_r]).to_event(2))
    w_pos = jax.nn.softplus(W_uncon) + 1e-8  # (B_t, B_r)

    # Precompute spatial basis per pair (N,N,B_r)
    D = pairwise_dists(node_xy)
    Psi_r = jnp.stack([gauss_bump(D, c, dist_scale) for c in dist_centers], axis=-1)

    # For each pair (i,j), collect coefficients along time basis:
    # S_t[i,j,:] = Σ_b w_pos[:, b] * Psi_r[i,j,b]  -> shape (N,N,B_t)
    S_t = jnp.tensordot(Psi_r, w_pos, axes=[[2], [1]])  # (N,N,B_t)

    # Denominator per pair (unit time integral): Z_pair[i,j] = S_t[i,j,:]·I_inf
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
        # For fixed source column j, vector over targets i: J_vec[i] = ∫ ψ̃(τ, r_{i,u_j}) dτ
        I_cap = jnp.stack([gauss_bump_int_0_to(cap, c, time_scale) for c in time_centers], axis=0)
        num_vec = jnp.dot(S_t[:, u_j, :], I_cap)  # (N,)
        J_vec = num_vec / denom[:, u_j]
        col = K_masked[:, u_j]
        col_sum = jnp.dot(col, J_vec)
        return carry + alpha * rowsum_MK[e_j] * col_sum, None

    exc_int, _ = lax.scan(comp_step, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))

    loglik = event_loglik - base_int - exc_int
    numpyro.factor("loglik", loglik)


# --------------- Enhanced joint kernel utilities ---------------
def reconstruct_joint_kernel(W_uncon, time_centers, time_scale, dist_centers, dist_scale, 
                           tau_grid, r_grid):
    """
    Reconstruct the joint spatio-temporal kernel ψ̃(τ, r) on a grid.
    
    Args:
        W_uncon: (B_t, B_r) unconstrained weights
        time_centers: (B_t,) temporal basis centers
        time_scale: temporal scale parameter
        dist_centers: (B_r,) distance basis centers  
        dist_scale: distance scale parameter
        tau_grid: (N_tau,) time lag grid
        r_grid: (N_r,) distance grid
    
    Returns:
        joint_kernel: (N_r, N_tau) kernel values
    """
    B_t, B_r = W_uncon.shape
    N_r, N_tau = len(r_grid), len(tau_grid)
    
    # Convert to numpy for easier computation
    W_uncon = np.asarray(W_uncon)
    time_centers = np.asarray(time_centers)
    dist_centers = np.asarray(dist_centers)
    
    # Reconstruct joint kernel values
    joint_kernel = np.zeros((N_r, N_tau))
    for i, r in enumerate(r_grid):
        for j, tau in enumerate(tau_grid):
            # Spatial basis
            phi_r = np.exp(-0.5 * ((r - dist_centers) / dist_scale) ** 2)
            # Temporal basis  
            phi_tau = np.exp(-0.5 * ((tau - time_centers) / time_scale) ** 2)
            # Joint kernel: ψ̃(τ, r) = Σ_{b_t, b_r} w[b_t, b_r] * φ_t(τ) * φ_r(r)
            joint_kernel[i, j] = np.sum(W_uncon * np.outer(phi_tau, phi_r))
    
    return joint_kernel


def compute_kernel_statistics(W_uncon, time_centers, time_scale, dist_centers, dist_scale):
    """
    Compute various statistics of the joint kernel.
    
    Returns:
        dict with kernel statistics
    """
    # Convert to numpy
    W_uncon = np.asarray(W_uncon)
    time_centers = np.asarray(time_centers)
    dist_centers = np.asarray(dist_centers)
    
    # Basic weight statistics
    weight_stats = {
        'mean': float(np.mean(W_uncon)),
        'std': float(np.std(W_uncon)),
        'min': float(np.min(W_uncon)),
        'max': float(np.max(W_uncon)),
        'sparsity': float(np.mean(W_uncon < 1e-6))  # Fraction of near-zero weights
    }
    
    # Temporal and spatial basis statistics
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
    
    # Interaction patterns
    interaction_stats = {
        'temporal_dominance': float(np.mean(np.abs(W_uncon), axis=1).tolist()),  # Per time basis
        'spatial_dominance': float(np.mean(np.abs(W_uncon), axis=0).tolist()),  # Per distance basis
        'cross_correlation': float(np.corrcoef(W_uncon.flatten(), 
                                             np.arange(W_uncon.size))[0, 1])
    }
    
    return {
        'weights': weight_stats,
        'temporal_basis': time_stats,
        'spatial_basis': dist_stats,
        'interactions': interaction_stats
    }


def analyze_kernel_behavior(W_uncon, time_centers, time_scale, dist_centers, dist_scale,
                          node_xy, tau_test=None, r_test=None):
    """
    Analyze the behavior of the joint kernel in the context of the network.
    
    Args:
        node_xy: (N, 2) node coordinates
        tau_test: test time lags
        r_test: test distances
    
    Returns:
        dict with behavioral analysis
    """
    if tau_test is None:
        tau_test = np.linspace(0, 2*time_centers.max(), 50)
    if r_test is None:
        r_test = np.linspace(0, 2*dist_centers.max(), 50)
    
    # Reconstruct kernel on test grid
    joint_kernel = reconstruct_joint_kernel(W_uncon, time_centers, time_scale,
                                          dist_centers, dist_scale, tau_test, r_test)
    
    # Compute pairwise distances in network
    N = node_xy.shape[0]
    D_network = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D_network[i, j] = np.sqrt(np.sum((node_xy[i] - node_xy[j])**2))
    
    # Distance distribution analysis
    dist_analysis = {
        'network_distances': D_network.flatten().tolist(),
        'mean_distance': float(np.mean(D_network)),
        'max_distance': float(np.max(D_network)),
        'distance_quantiles': np.percentile(D_network, [25, 50, 75]).tolist()
    }
    
    # Kernel behavior at network-relevant distances
    kernel_behavior = {}
    for r in [0.1, 0.5, 1.0, 2.0]:  # Test distances
        if r <= r_test.max():
            r_idx = np.argmin(np.abs(r_test - r))
            kernel_behavior[f'r_{r}'] = {
                'temporal_decay': joint_kernel[r_idx, :].tolist(),
                'peak_time': float(tau_test[np.argmax(joint_kernel[r_idx, :])]),
                'total_excitation': float(np.trapz(joint_kernel[r_idx, :], tau_test))
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
    p = argparse.ArgumentParser(description="Joint spatio-temporal nonparametric Hawkes with finite window W and unit-time-integral kernel per pair.")
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

    # Inference
    if args.method == "mcmc":
        init_strategy = None
        if args.svi_iters and args.svi_iters > 0:
            guide_warm = autoguide.AutoDelta(hawkes_np_st_model)
            svi_warm = SVI(hawkes_np_st_model, guide_warm, numpyro.optim.Adam(args.svi_lr), loss=Trace_ELBO())
            state = svi_warm.init(
                jax.random.PRNGKey(args.seed),
                t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                dist_centers=dist_centers, dist_scale=dist_scale,
                start_idx=start_idx, L_max=L_max, W=W_jax,
                N=N, M=M,
            )
            for _ in range(args.svi_iters):
                state, _ = svi_warm.update(
                    state,
                    t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                    time_centers=time_centers, time_scale=time_scale,
                    dist_centers=dist_centers, dist_scale=dist_scale,
                    start_idx=start_idx, L_max=L_max, W=W_jax,
                    N=N, M=M,
                )
            init_params = guide_warm.median(svi_warm.get_params(state))
            init_strategy = init_to_value(values=init_params)
            print(f"Finished SVI warmup: {args.svi_iters} iters. Starting MCMC...")
        kernel = NUTS(hawkes_np_st_model, target_accept_prob=0.85, init_strategy=init_strategy) if init_strategy else NUTS(hawkes_np_st_model, target_accept_prob=0.85)
        mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains, chain_method="parallel")
        mcmc.run(
            key,
            t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            dist_centers=dist_centers, dist_scale=dist_scale,
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
            "mcmc_state_np3.npz",
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
        print("Saved full MCMC posterior to mcmc_state_np3.npz")

    else:
        guide = autoguide.AutoDelta(hawkes_np_st_model)
        svi = SVI(hawkes_np_st_model, guide, numpyro.optim.Adam(args.svi_lr), loss=Trace_ELBO())
        state = svi.init(
            jax.random.PRNGKey(args.seed),
            t=t, u=u, e=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            dist_centers=dist_centers, dist_scale=dist_scale,
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

    # Save posterior means (no kernel grid reconstruction here; reconstruct as needed)
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
        "kernel_param": np.asarray(W_uncon_hat),  # (B_t,B_r) unconstrained
        "mcmc_state_file": "mcmc_state_np3.npz" if args.method == "mcmc" else None,
    }

    # Add enhanced joint kernel analysis
    print("Computing enhanced joint kernel analysis...")
    try:
        # Compute kernel statistics
        kernel_stats = compute_kernel_statistics(W_uncon_hat, time_centers, time_scale, 
                                              dist_centers, dist_scale)
        out["kernel_statistics"] = kernel_stats
        
        # Analyze kernel behavior in network context
        kernel_behavior = analyze_kernel_behavior(W_uncon_hat, time_centers, time_scale,
                                               dist_centers, dist_scale, node_locations)
        out["kernel_behavior"] = kernel_behavior
        
        print("  ✓ Kernel statistics computed")
        print("  ✓ Kernel behavior analyzed")
        
        # Print summary statistics
        print(f"\n=== JOINT KERNEL SUMMARY ===")
        print(f"Temporal bases: {len(time_centers)} (scale: {time_scale:.3g})")
        print(f"Distance bases: {len(dist_centers)} (scale: {dist_scale:.3g})")
        print(f"Weight statistics:")
        print(f"  Mean: {kernel_stats['weights']['mean']:.4f}")
        print(f"  Std:  {kernel_stats['weights']['std']:.4f}")
        print(f"  Sparsity: {kernel_stats['weights']['sparsity']:.1%}")
        print(f"Network distance analysis:")
        print(f"  Mean: {kernel_behavior['distance_analysis']['mean_distance']:.3f}")
        print(f"  Max:  {kernel_behavior['distance_analysis']['max_distance']:.3f}")
        print("==============================\n")
        
    except Exception as e:
        print(f"  ⚠ Enhanced analysis failed: {e}")
        out["kernel_statistics"] = None
        out["kernel_behavior"] = None

    with open(f"inference_result_np3_{args.data.split('.')[0]}.pickle", "wb") as f:
        pickle.dump(out, f)
    print(f"Saved posterior means to inference_result_np3_{args.data.split('.')[0]}.pickle")


if __name__ == "__main__":
    main() 