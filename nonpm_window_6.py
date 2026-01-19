#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network-Constrained Semiparametric Hawkes Model (matching thesis specification).

This implements the intensity function from the thesis:
    λ_u(t) = μ_u + α Σ_{i: t_i<t} g_t(t-t_i) · K_{x_i, u}

Where:
    - μ_u: baseline intensity for node u
    - α: global excitation amplitude (scalar)
    - g_t(τ): semiparametric temporal kernel (convex mixture of B_t Gaussian basis functions)
    - K_{uv}: spatial coupling based on distance with B_s Gaussian basis functions

Temporal kernel (§4.1):
    g_t(τ) = Σ_{b=1}^{B_t} w_b ψ_b(τ),  w_b ≥ 0, Σw_b = 1, ∫g_t(τ)dτ = 1

Spatial kernel (§4.2):
    K̂_{uv} = R^{(L)}_{uv} · (Σ_{r=1}^{B_s} β_r χ_r(d(u,v))),  β_r ≥ 0
    K_{uv} = K̂_{uv} / Σ_{v'} K̂_{uv'}  (row-normalized for identifiability)

Implementation uses column-normalization due to transpose convention (row=target, col=source).
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
    """Compute multi-hop reachability mask from adjacency matrix."""
    A = (adjacency > 0).astype(np.int32)
    N = A.shape[0]
    R = np.eye(N, dtype=np.int32)
    cur = A.copy()
    for _ in range(num_hops):
        R = (R | (cur > 0).astype(np.int32)).astype(np.int32)
        cur = (cur @ A > 0).astype(np.int32)
    return R.astype(np.float32)


def prep_events_structured(events, num_event_types=None):
    """Extract t, u, e arrays from events dict."""
    t = np.asarray(events["t"])
    u = np.asarray(events["u"])
    e = np.asarray(events["e"])
    T = float(t.max()) if t.size > 0 else 0.0
    N = int(u.max()) + 1 if u.size > 0 else 0
    M = int(num_event_types) if num_event_types is not None else (int(e.max()) + 1 if e.size > 0 else 1)
    return t, u, e, T, N, M


def pairwise_dists(xy):
    """Compute pairwise Euclidean distances from node coordinates."""
    # xy: (N, 2) array of node coordinates
    diff = xy[:, None, :] - xy[None, :, :]  # (N, N, 2)
    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)  # (N, N)


# ---------------- Gaussian basis utilities ----------------
def make_centers(width, n):
    """Create evenly spaced centers for Gaussian basis."""
    if n == 1:
        return jnp.array([0.5 * width])
    return jnp.linspace(0.0, width, n)


def gauss_bump(x, c, scale):
    """Gaussian bump function χ_r(d) = exp(-0.5 * ((d - c_r)/σ)^2)."""
    z = (x - c) / scale
    return jnp.exp(-0.5 * z * z)


def gauss_bump_int_0_to(x, c, scale):
    """∫_0^x exp(-0.5 * ((t - c)/s)^2) dt"""
    rt2 = jnp.sqrt(2.0)
    pref = scale * jnp.sqrt(jnp.pi / 2.0)
    return pref * (erf((x - c) / (rt2 * scale)) - erf((-c) / (rt2 * scale)))


def gauss_bump_int_0_to_inf(c, scale):
    """∫_0^∞ exp(-0.5 * ((t - c)/s)^2) dt"""
    rt2 = jnp.sqrt(2.0)
    return scale * jnp.sqrt(jnp.pi / 2.0) * (1.0 - erf((-c) / (rt2 * scale)))


# ---------------- Model ----------------
def hawkes_semiparametric_model(
    t, u, e, T,
    reach_mask,
    node_xy,
    time_centers, time_scale,
    space_centers, space_scale,
    start_idx, L_max, W,
    N: int, M: int,
):
    """
    Semiparametric Hawkes Model matching thesis specification.
    
    Intensity for node v, mark k at time t:
        λ_{v,k}(t) = μ_{v,k} + α Σ_{u} Σ_{ℓ} K_{uv} [M_K]_{ℓk} Σ_{i: t_i<t, u_i=u, e_i=ℓ} g(t-t_i)
    
    - μ_{v,k}: baseline intensity for node v, mark k
    - α: global excitation amplitude (scalar)
    - K_{uv}: distance-based coupling (Σ β_r χ_r(d(u,v))) masked by reachability, column-normalized
    - M_K[ℓ,k]: mark kernel (how mark ℓ excites mark k), row-normalized
    - g(τ): temporal kernel (Σ w_b ψ_b(τ)) with unit integral
    """
    Kevents = t.shape[0]

    # ---- Base rates μ_{v,k} (positive via softplus) ----
    mu_uncon = numpyro.sample("mu_uncon", dist.Normal(0.0, 1.0).expand([N, M]).to_event(2))
    mu = numpyro.deterministic("mu", jax.nn.softplus(mu_uncon) + 1e-8)

    # ---- Spatial kernel K_{uv} from distance-based basis functions (§4.2) ----
    # K̂_{uv} = R^{(L)}_{uv} · (Σ_{r=1}^{B_s} β_r χ_r(d(u,v)))
    B_s = space_centers.shape[0]
    beta_uncon = numpyro.sample("beta_uncon", dist.Normal(0.0, 0.8).expand([B_s]).to_event(1))
    beta = jax.nn.softplus(beta_uncon) + 1e-8  # β_r ≥ 0
    
    # Compute pairwise distances
    D = pairwise_dists(node_xy)  # (N, N)
    
    # Evaluate spatial basis functions χ_r(d) at all distances
    chi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)  # (N, N, B_s)
    
    # K̂_{uv} = Σ_r β_r χ_r(d(u,v))
    K_hat = jnp.tensordot(chi, beta, axes=[-1, 0])  # (N, N)
    
    # Apply reachability mask R^{(L)}_{uv}
    K_masked = K_hat * reach_mask  # (N, N)
    
    # Column-normalize (implementation convention: col=source)
    # This ensures Σ_v K_{uv} = 1 for each source u (after transpose accounting)
    colsum_K = jnp.maximum(jnp.sum(K_masked, axis=0), 1e-12)
    K = numpyro.deterministic("K", K_masked / colsum_K[None, :])
    
    # Store the spatial basis weights for diagnostics
    numpyro.deterministic("beta", beta)

    # ---- Mark kernel M_K (positive, row-normalized) ----
    M_uncon = numpyro.sample("M_uncon", dist.Normal(0.0, 1.0).expand([M, M]).to_event(2))
    M_pos = jax.nn.softplus(M_uncon) + 1e-8
    rowsum_M = jnp.maximum(jnp.sum(M_pos, axis=1), 1e-12)
    M_K = numpyro.deterministic("M_K", M_pos / rowsum_M[:, None])

    # ---- Global excitation amplitude α ----
    alpha = numpyro.sample("alpha", dist.Beta(2.0, 4.0))  # Prior mean ~0.33

    # ---- Temporal kernel g(τ) with unit integral on [0, ∞) (§4.1) ----
    # g_t(τ) = Σ_{b=1}^{B_t} w_b ψ_b(τ)
    B_t = time_centers.shape[0]
    a_uncon = numpyro.sample("a_uncon", dist.Normal(0.0, 0.8).expand([B_t]).to_event(1))
    w_pos = jax.nn.softplus(a_uncon) + 1e-8
    
    # Compute normalization constant for unit integral
    ints = jnp.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers])
    Z_t = jnp.dot(w_pos, ints) + 1e-12
    mix_w = w_pos / Z_t
    numpyro.deterministic("mix_w", mix_w)

    def g_scalar(delta):
        """Temporal kernel g(τ) at scalar τ ≥ 0."""
        delta = jnp.maximum(delta, 0.0)
        phi = jnp.exp(-0.5 * ((delta - time_centers) / time_scale) ** 2)
        return jnp.dot(phi, mix_w)

    def G_int_vec(delta):
        """∫_0^δ g(τ) dτ for vector δ."""
        delta = jnp.clip(delta, a_min=0.0)
        Phi_int = jnp.stack([gauss_bump_int_0_to(delta, c, time_scale) for c in time_centers], axis=-1)
        return Phi_int @ mix_w

    # ---- Event log-likelihood ----
    def step_event(carry, i):
        t_i = t[i]
        u_i = u[i]  # node of event i (target)
        e_i = e[i]  # mark of event i
        start_i = start_idx[i]

        def body(acc, k):
            j = i - 1 - k
            valid = (j >= start_i) & (j >= 0)
            j_clamped = jnp.clip(j, 0, Kevents - 1)
            
            dt = t_i - t[j_clamped]
            valid = valid & (dt <= W) & (dt > 0)
            
            u_j = u[j_clamped]  # source node
            e_j = e[j_clamped]  # source mark
            
            g_val = g_scalar(dt)
            
            # Contribution: α * K[u_i, u_j] * M_K[e_j, e_i] * g(dt)
            contrib = alpha * K[u_i, u_j] * M_K[e_j, e_i] * g_val
            contrib = jnp.where(valid, contrib, jnp.array(0.0, dtype=t.dtype))
            
            return acc + contrib, None

        excite_sum, _ = lax.scan(body, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(L_max))
        
        lam_ie = mu[u_i, e_i] + excite_sum
        lam_ie = jnp.clip(lam_ie, a_min=1e-12)
        
        return carry + jnp.log(lam_ie), None

    event_loglik, _ = lax.scan(step_event, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))

    # ---- Compensator (integral of intensity) ----
    # For linear Hawkes:
    #   ∫_0^T Σ_v Σ_k λ_{v,k}(t) dt
    # = T * Σ_v Σ_k μ_{v,k}  (baseline)
    # + α * Σ_j [Σ_v K[v, u_j]] * [Σ_k M_K[e_j, k]] * ∫_{t_j}^{min(T, t_j+W)} g(t - t_j) dt
    
    base_int = T * jnp.sum(mu)
    
    # For excitation: each event j contributes to compensator
    colsum_K_all = jnp.sum(K, axis=0)  # Σ_v K[v, u] for each source u
    rowsum_MK = jnp.sum(M_K, axis=1)    # Σ_k M_K[ℓ, k] for each source mark ℓ
    
    tail_limit = jnp.minimum(T - t, W)  # Integration limit for each event
    tail = G_int_vec(tail_limit)  # ∫_0^{tail_limit} g(τ) dτ for each event
    
    # Compensator contribution from excitation
    exc_int = alpha * jnp.sum(colsum_K_all[u] * rowsum_MK[e] * tail)

    loglik = event_loglik - base_int - exc_int
    numpyro.factor("loglik", loglik)


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser(
        description="Semiparametric Hawkes Model with distance-based spatial kernel"
    )
    p.add_argument("--data", type=str, required=True, help="Input pickle file")
    p.add_argument("--method", type=str, choices=["mcmc", "map"], default="mcmc")
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B_t", type=int, default=16, help="# temporal Gaussian basis functions")
    p.add_argument("--B_s", type=int, default=8, help="# spatial Gaussian basis functions")
    p.add_argument("--time_scale", type=float, default=None, help="Temporal basis width σ_t")
    p.add_argument("--space_scale", type=float, default=None, help="Spatial basis width σ_s")
    p.add_argument("--window", type=float, default=10.0, help="Finite look-back window W")
    p.add_argument("--svi_iters", type=int, default=0, help="SVI warmup iterations before MCMC")
    p.add_argument("--svi_lr", type=float, default=5e-2, help="SVI learning rate")
    p.add_argument("--num-hops", type=int, default=None, help="Override num_hops for reachability")
    args = p.parse_args()

    enable_x64()

    # Load data
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    events = data["events"]
    num_nodes = int(data["num_nodes"])
    num_event_types = int(data["num_event_types"])
    adjacency = np.asarray(data["adjacency_matrix"], dtype=float)
    num_hops = args.num_hops if args.num_hops is not None else int(data.get("num_hops", 1))
    print(f"Using num_hops: {num_hops}")

    t_np, u_np, e_np, T_np, N_ev, M_ev = prep_events_structured(events, num_event_types)
    assert num_nodes == N_ev and num_event_types == M_ev, f"Mismatch: {num_nodes} vs {N_ev}, {num_event_types} vs {M_ev}"

    # Sort by time
    order = np.argsort(t_np)
    t_np = t_np[order]
    u_np = u_np[order]
    e_np = e_np[order]

    reach_mask_np = compute_reachability(adjacency, num_hops=num_hops)

    # Get node coordinates (required for distance-based kernel)
    if "node_positions" in data:
        node_xy_np = np.asarray(data["node_positions"], dtype=np.float64)
    elif "node_xy" in data:
        node_xy_np = np.asarray(data["node_xy"], dtype=np.float64)
    elif "node_locations" in data:
        node_xy_np = np.asarray(data["node_locations"], dtype=np.float64)
    else:
        raise ValueError("Data must contain 'node_positions', 'node_xy', or 'node_locations' for distance-based spatial kernel")

    # Window and start indices
    W = args.window
    t_jnp = jnp.array(t_np, dtype=jnp.float64)
    u_jnp = jnp.array(u_np, dtype=jnp.int32)
    e_jnp = jnp.array(e_np, dtype=jnp.int32)
    
    # Compute start indices for window lookback
    start_idx_np = np.searchsorted(t_np, t_np - W, side='left')
    L_max = int(np.max(np.arange(len(t_np)) - start_idx_np + 1))
    print(f"Max lookback window: {L_max} events")

    # Time centers for temporal kernel
    B_t = args.B_t
    time_centers_np = make_centers(W, B_t)
    time_scale = args.time_scale if args.time_scale is not None else (W / (B_t - 1) if B_t > 1 else W / 2)

    # Space centers for spatial kernel (based on distance quantiles)
    B_s = args.B_s
    dists_flat = np.sqrt(np.sum((node_xy_np[:, None, :] - node_xy_np[None, :, :]) ** 2, axis=-1)).flatten()
    dists_nonzero = dists_flat[dists_flat > 0]
    if len(dists_nonzero) > 0:
        space_max = np.percentile(dists_nonzero, 95)
    else:
        space_max = 1.0
    space_centers_np = np.linspace(0.0, space_max, B_s)
    space_scale = args.space_scale if args.space_scale is not None else (space_max / (B_s - 1) if B_s > 1 else space_max / 2)
    
    print(f"Temporal: B_t={B_t}, time_scale={time_scale:.4f}, W={W}")
    print(f"Spatial: B_s={B_s}, space_scale={space_scale:.4f}, max_dist={space_max:.4f}")

    # Build model args
    model_args = (
        t_jnp, u_jnp, e_jnp, T_np,
        jnp.array(reach_mask_np, dtype=jnp.float32),
        jnp.array(node_xy_np, dtype=jnp.float64),
        jnp.array(time_centers_np, dtype=jnp.float64),
        time_scale,
        jnp.array(space_centers_np, dtype=jnp.float64),
        space_scale,
        jnp.array(start_idx_np, dtype=jnp.int32),
        L_max,
        W,
        num_nodes,
        num_event_types,
    )

    rng_key = jax.random.PRNGKey(args.seed)

    # Optional SVI warmup for initialization
    init_params = None
    if args.svi_iters > 0:
        print(f"\n--- SVI Warmup ({args.svi_iters} iterations) ---")
        guide = autoguide.AutoNormal(hawkes_semiparametric_model)
        opt = numpyro.optim.Adam(step_size=args.svi_lr)
        svi = SVI(hawkes_semiparametric_model, guide, opt, loss=Trace_ELBO())
        
        svi_result = svi.run(rng_key, args.svi_iters, *model_args, progress_bar=True)
        params = svi_result.params
        
        # Extract point estimates for MCMC initialization
        init_params = {}
        for k in params:
            if k.endswith("_auto_loc"):
                name = k.replace("_auto_loc", "")
                init_params[name] = params[k]
        print(f"SVI final loss: {svi_result.losses[-1]:.2f}")

    # MCMC
    print(f"\n--- MCMC ({args.warmup} warmup, {args.samples} samples, {args.chains} chains) ---")
    
    if init_params:
        init_strategy = init_to_value(values=init_params)
        kernel = NUTS(hawkes_semiparametric_model, init_strategy=init_strategy)
    else:
        kernel = NUTS(hawkes_semiparametric_model)
    
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains)
    mcmc.run(rng_key, *model_args)
    mcmc.print_summary()

    # Save results
    samples = mcmc.get_samples()
    data_prefix = args.data.replace(".pickle", "").replace("_events", "")
    
    # Include window in filename to distinguish different runs
    window_str = f"_w{W:.1f}".replace(".", "p")  # e.g., _w0p5, _w1p0, _w1p5
    
    # Full inference result pickle
    result = {
        "samples": {k: np.asarray(v) for k, v in samples.items()},
        "data_pickle": args.data,
        "mcmc_state_file": f"mcmc_state_np_{data_prefix}{window_str}.npz",
        "num_nodes": num_nodes,
        "num_event_types": num_event_types,
        "B_t": B_t,
        "B_s": B_s,
        "time_scale": time_scale,
        "space_scale": space_scale,
        "window": W,
        "num_hops": num_hops,
        "time_centers": np.asarray(time_centers_np),
        "space_centers": np.asarray(space_centers_np),
    }
    
    out_pickle = f"inference_result_np_{data_prefix}{window_str}_linear.pickle"
    with open(out_pickle, "wb") as f:
        pickle.dump(result, f)
    print(f"\n✓ Saved inference result to {out_pickle}")

    # Save MCMC state for visualization
    npz_out = f"mcmc_state_np_{data_prefix}{window_str}.npz"
    np.savez(npz_out, **{k: np.asarray(v) for k, v in samples.items()})
    print(f"✓ Saved MCMC samples to {npz_out}")


if __name__ == "__main__":
    main()
