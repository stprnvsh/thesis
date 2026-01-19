#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network-Constrained Marked Hawkes Model (matching the target intensity exactly).

This implements the intensity function:
    λ_{v,k}(t) = μ_{v,k} + α Σ_{u∈V} Σ_{ℓ∈M} K_{uv} [M_K]_{ℓk} Σ_{i: t_i<t} g(t-t_i) 𝟙{u_i=u, e_i=ℓ}

Key differences from nonpm_window_3/4/5:
- NO spatial kernel κ̃(r) or ψ̃(τ,r) - the network coupling K_{uv} directly encodes structure
- Only a temporal kernel g(τ) with unit integral
- K_{uv} is constrained by the reachability mask (multi-hop adjacency)
- M_K is the mark kernel matrix for cross-mark excitation

This avoids the identifiability issues caused by having both K and a spatial kernel.
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


# ---------------- Gaussian basis utilities ----------------
def make_centers(width, n):
    """Create evenly spaced centers for Gaussian basis."""
    if n == 1:
        return jnp.array([0.5 * width])
    return jnp.linspace(0.0, width, n)


def gauss_bump(x, c, scale):
    """Gaussian bump function."""
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
def hawkes_network_model(
    t, u, e, T,
    reach_mask,
    time_centers, time_scale,
    start_idx, L_max, W,
    N: int, M: int,
    nonlinearity: str = "linear",
    grid_times: jnp.ndarray = None,
    grid_starts: jnp.ndarray = None,
    grid_ends: jnp.ndarray = None,
    Lg_max: int = 0,
):
    """
    Network-Constrained Marked Hawkes Model.
    
    Intensity for node v, mark k at time t:
        λ_{v,k}(t) = φ(μ_{v,k} + α Σ_{u} Σ_{ℓ} K_{uv} [M_K]_{ℓk} Σ_{i: t_i<t, u_i=u, e_i=ℓ} g(t-t_i))
    
    - μ_{v,k}: baseline intensity for node v, mark k
    - α: global excitation amplitude (scalar)
    - K_{uv}: network coupling from u to v (constrained by reachability)
    - M_K[ℓ,k]: mark kernel (how mark ℓ excites mark k)
    - g(τ): temporal kernel with unit integral ∫g(τ)dτ = 1
    - φ: nonlinear link function (linear, softplus, relu, exp, power2)
    
    No spatial kernel - the network structure K encodes all spatial information.
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

    # ---- Network coupling K_{uv} (positive, masked by reachability) ----
    K_uncon = numpyro.sample("K_uncon", dist.Normal(0.0, 1.0).expand([N, N]).to_event(2))
    K_pos = jax.nn.softplus(K_uncon)
    K_pre = K_pos * reach_mask  # Zero out unreachable pairs
    # Normalize columns so each source sums to 1 (or nearly so)
    colsum_K = jnp.maximum(jnp.sum(K_pre, axis=0), 1e-12)
    K = numpyro.deterministic("K", K_pre / colsum_K[None, :])

    # ---- Mark kernel M_K (positive, row-normalized) ----
    M_uncon = numpyro.sample("M_uncon", dist.Normal(0.0, 1.0).expand([M, M]).to_event(2))
    M_pos = jax.nn.softplus(M_uncon) + 1e-8
    rowsum_M = jnp.maximum(jnp.sum(M_pos, axis=1), 1e-12)
    M_K = numpyro.deterministic("M_K", M_pos / rowsum_M[:, None])

    # ---- Global excitation amplitude α ----
    alpha = numpyro.sample("alpha", dist.Beta(2.0, 4.0))  # Prior mean ~0.33

    # ---- Temporal kernel g(τ) with unit integral on [0, ∞) ----
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
        u_i = u[i]  # node of event i
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
        
        eta_ie = mu[u_i, e_i] + excite_sum
        lam_ie = phi_link(eta_ie)
        
        return carry + jnp.log(lam_ie), None

    event_loglik, _ = lax.scan(step_event, init=jnp.array(0.0, dtype=t.dtype), xs=jnp.arange(Kevents))

    # ---- Compensator (integral of intensity) ----
    if nonlinearity == "linear":
        # Analytic compensator for linear Hawkes:
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
    
    else:
        # Numeric compensator on a time grid for nonlinear φ(η)
        def compensator_grid():
            Gg = grid_times.shape[0]
            if (Gg == 0) | (Lg_max == 0):
                # Baseline-only case
                return T * jnp.sum(phi_link(mu))

            def step_grid(carry, g):
                tg = grid_times[g]
                start_g = grid_starts[g]
                end_g = grid_ends[g]

                # Compute excitation at each (node, mark) pair
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

                    # K[:, u_j] gives how u_j affects each node v
                    colvec = K[:, u_j]  # (N,)
                    # M_K[e_j, :] gives how mark e_j affects each mark k
                    outer_em = (colvec[:, None]) * (M_K[e_j, :][None, :])  # (N, M)

                    excite_new = acc + (alpha * g_val) * outer_em
                    excite_mat = jnp.where(valid, excite_new, acc)
                    return excite_mat, None

                excite_mat, _ = lax.scan(body_j, init=excite_mat, xs=jnp.arange(Lg_max))

                eta = mu + excite_mat  # (N, M)
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


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser(
        description="Network-Constrained Marked Hawkes (no spatial kernel, pure temporal + network structure)"
    )
    p.add_argument("--data", type=str, required=True, help="Input pickle file")
    p.add_argument("--method", type=str, choices=["mcmc", "map"], default="mcmc")
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B_t", type=int, default=16, help="# temporal Gaussian basis functions")
    p.add_argument("--time_scale", type=float, default=None, help="Temporal basis width")
    p.add_argument("--window", type=float, default=10.0, help="Finite look-back window W")
    p.add_argument("--svi_iters", type=int, default=0, help="SVI warmup iterations before MCMC")
    p.add_argument("--svi_lr", type=float, default=5e-2, help="SVI learning rate")
    p.add_argument("--num-hops", type=int, default=None, help="Override num_hops for reachability")
    # Nonlinearity options
    p.add_argument("--nonlinearity", type=str,
                   choices=["linear", "softplus", "relu", "exp", "power2"],
                   default="linear", help="Link φ applied to intensity.")
    p.add_argument("--comp_grid", type=int, default=256,
                   help="Grid size for nonlinear compensator; will be unioned with event times.")
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

    # Window and start indices
    W = float(args.window) if args.window is not None else np.inf
    if np.isfinite(W):
        starts = np.searchsorted(t_np, t_np - W, side="left")
        starts = np.minimum(starts, np.arange(t_np.shape[0]))
    else:
        starts = np.zeros_like(t_np, dtype=np.int64)
    L_max = int(np.max(np.arange(t_np.shape[0]) - starts)) if t_np.size else 0

    print(f"Events: {len(t_np)}, Nodes: {num_nodes}, Marks: {num_event_types}")
    print(f"Time span: {T_np:.4f}, Window: {W}, L_max: {L_max}")

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
        print(f"Nonlinear compensator grid: {len(grid_times_np)} points, Lg_max: {Lg_max}")
    else:
        grid_times_np = np.array([], dtype=np.float64)
        grid_starts_np = np.array([], dtype=np.int64)
        grid_ends_np = np.array([], dtype=np.int64)
        Lg_max = 0

    # JAX arrays
    key = jax.random.PRNGKey(args.seed)
    t = jnp.asarray(t_np)
    u = jnp.asarray(u_np)
    e = jnp.asarray(e_np)
    T = jnp.asarray(T_np, dtype=t.dtype)
    reach_mask = jnp.asarray(reach_mask_np)
    start_idx = jnp.asarray(starts, dtype=jnp.int32)
    W_jax = jnp.asarray(W, dtype=t.dtype)

    grid_times = jnp.asarray(grid_times_np)
    grid_starts = jnp.asarray(grid_starts_np, dtype=jnp.int32)
    grid_ends = jnp.asarray(grid_ends_np, dtype=jnp.int32)
    Lg_max = int(Lg_max)

    N = int(num_nodes)
    M = int(num_event_types)

    # Temporal basis
    B_t = int(args.B_t)
    time_centers = make_centers(T, B_t)
    if args.time_scale is None:
        time_scale = (T / max(B_t - 1, 1)) * 1.25
    else:
        time_scale = float(args.time_scale)
    time_scale = jnp.asarray(time_scale, dtype=t.dtype)

    print(f"Temporal basis: {B_t} centers, scale={float(time_scale):.4f}")

    # Inference
    if args.method == "mcmc":
        init_strategy = None
        if args.svi_iters > 0:
            guide_warm = autoguide.AutoDelta(hawkes_network_model)
            svi_warm = SVI(hawkes_network_model, guide_warm, numpyro.optim.Adam(args.svi_lr), loss=Trace_ELBO())
            state = svi_warm.init(
                jax.random.PRNGKey(args.seed),
                t=t, u=u, e=e, T=T, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                start_idx=start_idx, L_max=L_max, W=W_jax,
                N=N, M=M,
                nonlinearity=args.nonlinearity,
                grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max,
            )
            for i in range(args.svi_iters):
                state, loss = svi_warm.update(
                    state,
                    t=t, u=u, e=e, T=T, reach_mask=reach_mask,
                    time_centers=time_centers, time_scale=time_scale,
                    start_idx=start_idx, L_max=L_max, W=W_jax,
                    N=N, M=M,
                    nonlinearity=args.nonlinearity,
                    grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max,
                )
                if (i + 1) % 200 == 0:
                    print(f"[SVI warmup] iter {i+1:04d} loss={float(loss):.3f}")
            init_params = guide_warm.median(svi_warm.get_params(state))
            init_strategy = init_to_value(values=init_params)
            print(f"Finished SVI warmup: {args.svi_iters} iters. Starting MCMC...")

        kernel = NUTS(hawkes_network_model, target_accept_prob=0.85, init_strategy=init_strategy) \
            if init_strategy else NUTS(hawkes_network_model, target_accept_prob=0.85)
        
        mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, 
                    num_chains=args.chains, chain_method="parallel")
        mcmc.run(
            key,
            t=t, u=u, e=e, T=T, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            start_idx=start_idx, L_max=L_max, W=W_jax,
            N=N, M=M,
            nonlinearity=args.nonlinearity,
            grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max,
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()

        mu_hat = jnp.mean(posterior["mu"], axis=0)
        K_hat = jnp.mean(posterior["K"], axis=0)
        M_K_hat = jnp.mean(posterior["M_K"], axis=0)
        alpha_hat = float(jnp.mean(posterior["alpha"]))
        mix_w_hat = jnp.mean(posterior["mix_w"], axis=0)

        # Save MCMC state (include nonlinearity in filename)
        model_name = args.nonlinearity if args.nonlinearity != "linear" else ""
        suffix = f"_{model_name}" if model_name else ""
        mcmc_file = f"mcmc_state_np6_{args.data.split('.')[0]}{suffix}.npz"
        np.savez(
            mcmc_file,
            mu=np.asarray(posterior["mu"]),
            K=np.asarray(posterior["K"]),
            M_K=np.asarray(posterior["M_K"]),
            alpha=np.asarray(posterior["alpha"]),
            a_uncon=np.asarray(posterior["a_uncon"]),
            mix_w=np.asarray(posterior["mix_w"]),
            time_centers=np.asarray(time_centers),
            time_scale=float(time_scale),
            t=np.asarray(t), u=np.asarray(u), e=np.asarray(e), T=float(T),
            reach_mask=np.asarray(reach_mask_np),
            start_idx=np.asarray(starts), L_max=L_max, window=W if np.isfinite(W) else np.inf,
            nonlinearity=args.nonlinearity,
        )
        print(f"Saved full MCMC posterior to {mcmc_file}")

    else:
        guide = autoguide.AutoDelta(hawkes_network_model)
        svi = SVI(hawkes_network_model, guide, numpyro.optim.Adam(args.svi_lr), loss=Trace_ELBO())
        state = svi.init(
            jax.random.PRNGKey(args.seed),
            t=t, u=u, e=e, T=T, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            start_idx=start_idx, L_max=L_max, W=W_jax,
            N=N, M=M,
            nonlinearity=args.nonlinearity,
            grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max,
        )
        svi_iters = args.svi_iters if args.svi_iters > 0 else 2000
        for i in range(svi_iters):
            state, loss = svi.update(
                state,
                t=t, u=u, e=e, T=T, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                start_idx=start_idx, L_max=L_max, W=W_jax,
                N=N, M=M,
                nonlinearity=args.nonlinearity,
                grid_times=grid_times, grid_starts=grid_starts, grid_ends=grid_ends, Lg_max=Lg_max,
            )
            if (i + 1) % 200 == 0:
                print(f"[SVI] iter {i+1:04d} loss={float(loss):.3f}")
        
        params_map = svi.get_params(state)
        mu_hat = params_map["mu"]
        K_hat = params_map["K"]
        M_K_hat = params_map["M_K"]
        alpha_hat = float(params_map["alpha"])
        mix_w_hat = params_map["mix_w"]
        mcmc_file = None

    # Save results
    print("\n=== Posterior Summary ===")
    print(f"α (excitation amplitude): {alpha_hat:.6f}")
    print(f"μ shape: {tuple(np.asarray(mu_hat).shape)}, mean: {float(np.mean(mu_hat)):.6f}")
    print(f"K shape: {tuple(np.asarray(K_hat).shape)}, mean: {float(np.mean(K_hat)):.6f}")
    print(f"M_K:\n{np.asarray(M_K_hat)}")
    print(f"nonlinearity: {args.nonlinearity}")

    # Create filename with nonlinearity info
    model_name = args.nonlinearity if args.nonlinearity != "linear" else ""
    suffix = f"_{model_name}" if model_name else ""

    out = {
        "mu_hat": np.asarray(mu_hat),
        "K_hat": np.asarray(K_hat),
        "M_K_hat": np.asarray(M_K_hat),
        "alpha_hat": float(alpha_hat),
        "mix_w_hat": np.asarray(mix_w_hat),
        "N": N,
        "M": M,
        "T": float(T),
        "reach_mask": np.asarray(reach_mask_np),
        "data_pickle": args.data,
        "method": args.method,
        "time_centers": np.asarray(time_centers),
        "time_scale": float(time_scale),
        "window": W if np.isfinite(W) else np.inf,
        "L_max": L_max,
        "mcmc_state_file": mcmc_file,
        "model_type": "network_constrained",  # Flag for visualization
        "nonlinearity": args.nonlinearity,
        "comp_grid": int(args.comp_grid),
    }

    out_file = f"inference_result_np6_{args.data.split('.')[0]}{suffix}.pickle"
    with open(out_file, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved posterior means to {out_file}")


if __name__ == "__main__":
    main()

