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
    nonlinearity: str = "linear",
    grid_times: jnp.ndarray = None,
    grid_starts: jnp.ndarray = None,
    grid_ends: jnp.ndarray = None,
    Lg_max: int = 0,
):
    """
    Semiparametric Hawkes Model matching thesis specification.
    
    Intensity for node v, mark k at time t:
        λ_{v,k}(t) = φ(μ_{v,k} + α Σ_{u} Σ_{ℓ} K_{uv} [M_K]_{ℓk} Σ_{i: t_i<t, u_i=u, e_i=ℓ} g(t-t_i))
    
    - μ_{v,k}: baseline intensity for node v, mark k
    - α: global excitation amplitude (scalar)
    - K_{uv}: distance-based coupling (Σ β_r χ_r(d(u,v))) masked by reachability, column-normalized
    - M_K[ℓ,k]: mark kernel (how mark ℓ excites mark k), row-normalized
    - g(τ): temporal kernel (Σ w_b ψ_b(τ)) with unit integral
    - φ: nonlinear link function (linear, softplus, relu, exp, power2)
    """
    
    def phi_link(x):
        """Nonlinear link function φ applied to intensity."""
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

    # ---- Base rates μ_{v,k} (positive via softplus) ----
    mu_uncon = numpyro.sample("mu_uncon", dist.Normal(0.0, 1.0).expand([N, M]).to_event(2))
    mu = numpyro.deterministic("mu", jax.nn.softplus(mu_uncon) + 1e-8)

    # ---- Spatial kernel K_{uv} from distance-based basis functions (§4.2) ----
    B_s = space_centers.shape[0]
    beta_uncon = numpyro.sample("beta_uncon", dist.Normal(0.0, 0.8).expand([B_s]).to_event(1))
    beta = jax.nn.softplus(beta_uncon) + 1e-8
    
    D = pairwise_dists(node_xy)
    chi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)
    K_hat = jnp.tensordot(chi, beta, axes=[-1, 0])
    K_masked = K_hat * reach_mask
    colsum_K = jnp.maximum(jnp.sum(K_masked, axis=0), 1e-12)
    K = numpyro.deterministic("K", K_masked / colsum_K[None, :])
    numpyro.deterministic("beta", beta)

    # ---- Mark kernel M_K (positive, row-normalized) ----
    M_uncon = numpyro.sample("M_uncon", dist.Normal(0.0, 1.0).expand([M, M]).to_event(2))
    M_pos = jax.nn.softplus(M_uncon) + 1e-8
    rowsum_M = jnp.maximum(jnp.sum(M_pos, axis=1), 1e-12)
    M_K = numpyro.deterministic("M_K", M_pos / rowsum_M[:, None])

    # ---- Global excitation amplitude α ----
    alpha = numpyro.sample("alpha", dist.Beta(2.0, 4.0))

    # ---- Temporal kernel g(τ) with unit integral ----
    B_t = time_centers.shape[0]
    a_uncon = numpyro.sample("a_uncon", dist.Normal(0.0, 0.8).expand([B_t]).to_event(1))
    w_pos = jax.nn.softplus(a_uncon) + 1e-8
    ints = jnp.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers])
    Z_t = jnp.dot(w_pos, ints) + 1e-12
    mix_w = w_pos / Z_t
    numpyro.deterministic("mix_w", mix_w)

    def g_scalar(delta):
        delta = jnp.maximum(delta, 0.0)
        phi = jnp.exp(-0.5 * ((delta - time_centers) / time_scale) ** 2)
        return jnp.dot(phi, mix_w)

    def G_int_vec(delta):
        delta = jnp.clip(delta, a_min=0.0)
        Phi_int = jnp.stack([gauss_bump_int_0_to(delta, c, time_scale) for c in time_centers], axis=-1)
        return Phi_int @ mix_w

    # ---- Event log-likelihood ----
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
            valid = valid & (dt <= W) & (dt > 0)
            u_j = u[j_clamped]
            e_j = e[j_clamped]
            g_val = g_scalar(dt)
            contrib = alpha * K[u_i, u_j] * M_K[e_j, e_i] * g_val
            contrib = jnp.where(valid, contrib, jnp.array(0.0, dtype=t.dtype))
            return acc + contrib, None

        excite_sum, _ = lax.scan(body, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(L_max))
        eta_ie = mu[u_i, e_i] + excite_sum
        lam_ie = phi_link(eta_ie)
        return carry + jnp.log(lam_ie), None

    event_loglik, _ = lax.scan(step_event, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))

    # ---- Compensator ----
    if nonlinearity == "linear":
        base_int = T * jnp.sum(mu)
        colsum_K_all = jnp.sum(K, axis=0)
        rowsum_MK = jnp.sum(M_K, axis=1)
        tail_limit = jnp.minimum(T - t, W)
        tail = G_int_vec(tail_limit)
        exc_int = alpha * jnp.sum(colsum_K_all[u] * rowsum_MK[e] * tail)
        loglik = event_loglik - base_int - exc_int
    else:
        # Numeric compensator for nonlinear φ
        def compensator_grid():
            Gg = grid_times.shape[0]
            if (Gg == 0) | (Lg_max == 0):
                return T * jnp.sum(phi_link(mu))

            def step_grid(carry, g):
                tg = grid_times[g]
                start_g = grid_starts[g]
                end_g = grid_ends[g]
                excite_mat = jnp.zeros((N, M), dtype=t.dtype)

                def body_j(acc, k):
                    j = end_g - k
                    valid = (j >= start_g) & (j >= 0)
                    j = jnp.clip(j, 0, Kevents - 1)
                    dt = tg - t[j]
                    valid = valid & (dt <= W) & (dt >= 0.0)
                    u_j = u[j]
                    e_j = e[j]
                    g_val = g_scalar(dt)
                    colvec = K[:, u_j]
                    outer_em = (colvec[:, None]) * (M_K[e_j, :][None, :])
                    excite_new = acc + (alpha * g_val) * outer_em
                    return jnp.where(valid, excite_new, acc), None

                excite_mat, _ = lax.scan(body_j, init=excite_mat, xs=jnp.arange(Lg_max))
                eta = mu + excite_mat
                lam = phi_link(eta)
                return carry, jnp.sum(lam)

            _, lam_series = lax.scan(step_grid, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(grid_times.shape[0]))
            dt = jnp.diff(grid_times)
            avg = 0.5 * (lam_series[:-1] + lam_series[1:])
            return jnp.sum(avg * dt)

        comp_int = compensator_grid()
        loglik = event_loglik - comp_int

    numpyro.factor("loglik", loglik)


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser(description="Semiparametric Hawkes Model with distance-based spatial kernel")
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
    p.add_argument("--nonlinearity", type=str, choices=["linear", "softplus", "relu", "exp", "power2"],
                   default="linear", help="Link φ applied to intensity.")
    p.add_argument("--comp_grid", type=int, default=256, help="Grid size for nonlinear compensator")
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
    assert num_nodes == N_ev and num_event_types == M_ev

    order = np.argsort(t_np)
    t_np, u_np, e_np = t_np[order], u_np[order], e_np[order]

    reach_mask_np = compute_reachability(adjacency, num_hops=num_hops)

    # Get node coordinates
    if "node_positions" in data:
        node_xy_np = np.asarray(data["node_positions"], dtype=np.float64)
    elif "node_xy" in data:
        node_xy_np = np.asarray(data["node_xy"], dtype=np.float64)
    elif "node_locations" in data:
        node_xy_np = np.asarray(data["node_locations"], dtype=np.float64)
    else:
        raise ValueError("Data must contain node coordinates")

    W = args.window
    start_idx_np = np.searchsorted(t_np, t_np - W, side='left')
    L_max = int(np.max(np.arange(len(t_np)) - start_idx_np + 1))
    print(f"Max lookback window: {L_max} events")

    # Temporal basis
    B_t = args.B_t
    time_centers_np = make_centers(W, B_t)
    time_scale = args.time_scale if args.time_scale else (W / (B_t - 1) if B_t > 1 else W / 2)

    # Spatial basis
    B_s = args.B_s
    dists_flat = np.sqrt(np.sum((node_xy_np[:, None, :] - node_xy_np[None, :, :]) ** 2, axis=-1)).flatten()
    dists_nonzero = dists_flat[dists_flat > 0]
    space_max = np.percentile(dists_nonzero, 95) if len(dists_nonzero) > 0 else 1.0
    space_centers_np = np.linspace(0.0, space_max, B_s)
    space_scale = args.space_scale if args.space_scale else (space_max / (B_s - 1) if B_s > 1 else space_max / 2)

    print(f"Temporal: B_t={B_t}, time_scale={time_scale:.4f}, W={W}")
    print(f"Spatial: B_s={B_s}, space_scale={space_scale:.4f}, max_dist={space_max:.4f}")
    print(f"Nonlinearity: {args.nonlinearity}")

    # Compensator grid for nonlinear
    if args.nonlinearity != "linear":
        G = max(int(args.comp_grid), 2)
        uniform = np.linspace(0.0, T_np, G)
        grid_times_np = np.unique(np.concatenate([uniform, t_np]))
        grid_starts_np = np.searchsorted(t_np, grid_times_np - W, side="left") if np.isfinite(W) else np.zeros_like(grid_times_np, dtype=np.int64)
        grid_ends_np = np.searchsorted(t_np, grid_times_np, side="left") - 1
        valid_len = np.maximum(grid_ends_np - grid_starts_np + 1, 0)
        Lg_max = int(valid_len.max()) if valid_len.size else 0
        print(f"Nonlinear compensator grid: {len(grid_times_np)} points, Lg_max: {Lg_max}")
    else:
        grid_times_np = np.array([], dtype=np.float64)
        grid_starts_np = np.array([], dtype=np.int64)
        grid_ends_np = np.array([], dtype=np.int64)
        Lg_max = 0

    # JAX arrays
    rng_key = jax.random.PRNGKey(args.seed)
    model_kwargs = dict(
        t=jnp.array(t_np, dtype=jnp.float64),
        u=jnp.array(u_np, dtype=jnp.int32),
        e=jnp.array(e_np, dtype=jnp.int32),
        T=T_np,
        reach_mask=jnp.array(reach_mask_np, dtype=jnp.float32),
        node_xy=jnp.array(node_xy_np, dtype=jnp.float64),
        time_centers=jnp.array(time_centers_np, dtype=jnp.float64),
        time_scale=time_scale,
        space_centers=jnp.array(space_centers_np, dtype=jnp.float64),
        space_scale=space_scale,
        start_idx=jnp.array(start_idx_np, dtype=jnp.int32),
        L_max=L_max,
        W=W,
        N=num_nodes,
        M=num_event_types,
        nonlinearity=args.nonlinearity,
        grid_times=jnp.array(grid_times_np, dtype=jnp.float64),
        grid_starts=jnp.array(grid_starts_np, dtype=jnp.int32),
        grid_ends=jnp.array(grid_ends_np, dtype=jnp.int32),
        Lg_max=Lg_max,
    )

    # MCMC
    print(f"\n--- MCMC ({args.warmup} warmup, {args.samples} samples, {args.chains} chains) ---")
    kernel = NUTS(hawkes_semiparametric_model, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains, chain_method="parallel")
    mcmc.run(rng_key, **model_kwargs)
    mcmc.print_summary()

    samples = mcmc.get_samples()
    data_prefix = args.data.replace(".pickle", "").replace("_events", "")
    
    # Filename with window and nonlinearity
    window_str = f"_w{W:.1f}".replace(".", "p")
    nonlin_str = f"_{args.nonlinearity}" if args.nonlinearity != "linear" else ""

    result = {
        "samples": {k: np.asarray(v) for k, v in samples.items()},
        "data_pickle": args.data,
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
        "nonlinearity": args.nonlinearity,
    }

    out_pickle = f"inference_result_np_{data_prefix}{window_str}{nonlin_str}.pickle"
    with open(out_pickle, "wb") as f:
        pickle.dump(result, f)
    print(f"\n✓ Saved inference result to {out_pickle}")

    npz_out = f"mcmc_state_np_{data_prefix}{window_str}{nonlin_str}.npz"
    np.savez(npz_out, **{k: np.asarray(v) for k, v in samples.items()})
    print(f"✓ Saved MCMC samples to {npz_out}")


if __name__ == "__main__":
    main()
