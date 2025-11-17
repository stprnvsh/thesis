#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marked multivariate Hawkes with nonparametric temporal & spatial kernels
(basis expansions with positive weights), and parametric K (node->node) and
M_K (mark->mark) as in your previous model.

- g(τ) = sum_k a_k * N(τ; c_k, ℓ_t^2), τ >= 0         (temporal kernel)
- κ(r) = sum_m b_m * N(r; s_m, ρ_s^2),   r >= 0         (spatial kernel)
- G_node = K_masked ∘ κ(distances)

Both kernels are fully continuous and learn arbitrary smooth shapes
(as basis size increases). The time integral of g has a closed form via erf.

WARNING: This model computes intensities in O(K^2). For very large K,
consider windowing in time, or a coarser basis with truncation.

Saves:
- Posterior means to inference_result_np.pickle
- Full posterior draws (thin-able) to mcmc_state_np.npz
- Optional: summary printing of true vs. inferred if available in the data pickle.
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


# -------------------------------------------------------------
# Platform setup (CPU defaults; change if you want GPU)
# -------------------------------------------------------------
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(10)


# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
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


def pairwise_dists(node_xy):  # (N,2) -> (N,N) distances
    diff = node_xy[:, None, :] - node_xy[None, :, :]
    d2 = jnp.sum(diff * diff, axis=-1)
    return jnp.sqrt(jnp.maximum(d2, 0.0))


# -------------------------------------------------------------
# Gaussian basis utilities (positive radial bumps)
# -------------------------------------------------------------
def make_centers(width, n):
    # equi-spaced centers in [0, width]
    if n == 1:
        return jnp.array([0.5 * width])
    xs = jnp.linspace(0.0, width, n)
    return xs


def gauss_bump(x, c, scale):
    # exp(-0.5 * ((x - c)/scale)^2)
    z = (x - c) / scale
    return jnp.exp(-0.5 * z * z)


def gauss_bump_int_0_to(x, c, scale):
    """
    ∫_0^x exp(-0.5 * ((t - c)/s)^2) dt  =  s * sqrt(pi/2) * [erf((x - c)/(sqrt(2)*s)) - erf((-c)/(sqrt(2)*s))]
    """
    rt2 = jnp.sqrt(2.0)
    pref = scale * jnp.sqrt(jnp.pi / 2.0)
    return pref * (erf((x - c) / (rt2 * scale)) - erf((-c) / (rt2 * scale)))


# -------------------------------------------------------------
# Nonparametric Hawkes model
# -------------------------------------------------------------
def hawkes_np_model(
    t_obs, u_obs, e_obs, T,
    node_xy, reach_mask,
    # basis hyperparameters (fixed design)
    time_centers, time_scale,
    space_centers, space_scale,
    N: int, M: int
):
    """
    Nonparametric temporal g(τ) and spatial κ(r) via Gaussian bump bases (positive weights).
    K and M_K are as before (positive via softplus with weak Normal priors).

    Intensity for event at (u_i, e_i, t_i):
      λ_{u_i,e_i}(t_i) = μ[u_i,e_i] + sum_{j: t_j < t_i} G_node[u_i, u_j] * M_K[e_j, e_i] * g(t_i - t_j)
    where G_node = K_masked ∘ κ(Distances).
    """
    # Priors on base rate (per node, per mark)
    mu_uncon = numpyro.sample("mu_uncon", dist.Normal(0.0, 1.0).expand([N, M]).to_event(2))
    mu = numpyro.deterministic("mu", jax.nn.softplus(mu_uncon) + 1e-8)  # (N,M) positive

    # Priors on K (node interaction, masked) and M_K (mark interaction)
    K_uncon = numpyro.sample("K_uncon", dist.Normal(0.0, 0.5).expand([N, N]).to_event(2))
    K_pos = jax.nn.softplus(K_uncon)
    K_masked = numpyro.deterministic("K_masked", K_pos * reach_mask)  # (N,N)

    M_uncon = numpyro.sample("M_uncon", dist.Normal(0.0, 0.5).expand([M, M]).to_event(2))
    M_K = numpyro.deterministic("M_K", jax.nn.softplus(M_uncon) + 1e-8)  # (M,M)

    # ---- Nonparametric spatial kernel κ(r) = sum_m b_m ψ_m(r)
    B_s = space_centers.shape[0]
    b_uncon = numpyro.sample("b_uncon", dist.Normal(0.0, 0.5).expand([B_s]).to_event(1))
    b = jax.nn.softplus(b_uncon) + 1e-8  # positive weights

    # pairwise distances and κ matrix
    D = pairwise_dists(node_xy)  # (N,N)
    # Evaluate basis on all distances then weight
    psi = jnp.stack([gauss_bump(D, c, space_scale) for c in space_centers], axis=-1)  # (N,N,B_s)
    kappa = numpyro.deterministic("kappa", jnp.tensordot(psi, b, axes=[-1, 0]))       # (N,N)
    G_node = numpyro.deterministic("G_node", K_masked * kappa)                         # (N,N)

    # ---- Nonparametric temporal kernel g(τ) = sum_k a_k φ_k(τ), τ>=0
    B_t = time_centers.shape[0]
    a_uncon = numpyro.sample("a_uncon", dist.Normal(0.0, 0.5).expand([B_t]).to_event(1))
    a = jax.nn.softplus(a_uncon) + 1e-8  # positive weights

    # Helper: evaluate g(Δ) and its integral G(Δ) for Δ>=0
    def g_eval(delta):  # (K,) -> (K,)
        delta = jnp.clip(delta, a_min=0.0)
        phi = jnp.stack([gauss_bump(delta, c, time_scale) for c in time_centers], axis=-1)  # (K,B_t)
        return phi @ a  # (K,)

    def G_int(delta):  # ∫_0^Δ g(τ)dτ
        delta = jnp.clip(delta, a_min=0.0)
        Phi_int = jnp.stack([gauss_bump_int_0_to(delta, c, time_scale) for c in time_centers], axis=-1)  # (K,B_t)
        return Phi_int @ a  # (K,)

    # Sort events by time (required for proper masking)
    order = jnp.argsort(t_obs)
    t = t_obs[order]
    u = u_obs[order]
    e = e_obs[order]
    Kevents = t.shape[0]

    # Build indices for vectorized summation against all *previous* events
    # For each i, we will compute contributions from all j using a mask j < i.
    # This is O(K^2) but simple/stable in JAX.
    def step(carry, i):
        # i-th event
        t_i = t[i]
        u_i = u[i]
        e_i = e[i]

        # Δ_i - t of all events (including future); mask for past-only
        delta = t_i - t                      # (K,)
        mask = (delta > 0.0).astype(t.dtype) # 1 for j<i, else 0

        # temporal kernel on all deltas (zeros out for future by mask)
        g_vec = g_eval(delta) * mask         # (K,)

        # node/mark couplings for all sources j toward (u_i,e_i)
        # G_row[u_i, u_j]  and  M_K[e_j, e_i]
        G_row = G_node[u_i, u]               # (K,)
        MK_col = M_K[e, e_i]                 # (K,)

        lam_ie = mu[u_i, e_i] + jnp.sum(G_row * MK_col * g_vec)
        lam_ie = jnp.clip(lam_ie, a_min=1e-12)

        return carry + jnp.log(lam_ie), None

    event_loglik, _ = lax.scan(step, init=jnp.array(0.0), xs=jnp.arange(Kevents))

    # ---- Integral term (baseline + excitation integrals)
    base_int = T * jnp.sum(mu)

    # For each past event j at (u_j, e_j, t_j): contribution:
    #   [sum_u G_node[u, u_j]] * [sum_e M_K[e_j, e]] * ∫_0^{T - t_j} g(τ) dτ
    colsum_G = jnp.sum(G_node, axis=0)  # (N,) sums over u -> indexed by source node u_j
    rowsum_MK = jnp.sum(M_K, axis=1)    # (M,) sums over target marks -> indexed by source mark e_j

    tail = G_int(T - t)                  # (K,)
    exc_int = jnp.sum(colsum_G[u] * rowsum_MK[e] * tail)

    loglik = event_loglik - base_int - exc_int
    numpyro.factor("loglik", loglik)


# -------------------------------------------------------------
# Main driver
# -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Nonparametric kernels (temporal & spatial) Hawkes with NumPyro.")
    p.add_argument("--data", type=str, default="traffic_hawkes_simulation2.pickle")
    p.add_argument("--method", type=str, choices=["mcmc", "map"], default="mcmc")
    p.add_argument("--warmup", type=int, default=3000)
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # basis choices
    p.add_argument("--B_t", type=int, default=10, help="# temporal Gaussian basis")
    p.add_argument("--B_s", type=int, default=10, help="# spatial Gaussian basis")
    p.add_argument("--time_scale", type=float, default=None, help="temporal basis width; default auto from T")
    p.add_argument("--space_scale", type=float, default=None, help="spatial basis width; default auto from max dist")
    args = p.parse_args()

    enable_x64()

    # --- Load generator pickle (same format you use)
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    events = data["events"]
    num_nodes = int(data["num_nodes"])
    num_event_types = int(data["num_event_types"])
    node_locations = np.asarray(data["node_locations"], dtype=float)  # (N,2)
    adjacency = np.asarray(data["adjacency_matrix"], dtype=float)     # (N,N)
    num_hops = int(data.get("num_hops", 1))
    params_init = data.get("params", None)
    # Optional "truth" for M_K in data["mark_kernel_matrix"]

    # --- Prepare arrays
    t_np, u_np, e_np, T_np, N_from_ev, M_from_ev = prep_events_structured(events, num_event_types)
    assert num_nodes == N_from_ev
    assert num_event_types == M_from_ev
    reach_mask_np = compute_reachability(adjacency, num_hops=num_hops)

    # --- JAX arrays
    key = jax.random.PRNGKey(args.seed)
    t = jnp.asarray(t_np)
    u = jnp.asarray(u_np)
    e = jnp.asarray(e_np)
    T = jnp.asarray(T_np, dtype=t.dtype)
    node_xy = jnp.asarray(node_locations)
    reach_mask = jnp.asarray(reach_mask_np)

    N = int(num_nodes)
    M = int(num_event_types)

    # --- Build basis designs
    # time centers in [0, T]
    B_t = int(args.B_t)
    time_centers = make_centers(T, B_t)
    if args.time_scale is None:
        time_scale = (T / max(B_t - 1, 1)) * 1.25  # a bit wider than spacing
    else:
        time_scale = float(args.time_scale)
    time_scale = jnp.asarray(time_scale)

    # spatial centers in [0, r_max]
    # r_max from node pair distances
    D_np = np.sqrt(np.maximum(((node_locations[:, None, :] - node_locations[None, :, :]) ** 2).sum(-1), 0.0))
    r_max = float(D_np.max()) if D_np.size > 0 else 1.0
    B_s = int(args.B_s)
    space_centers = make_centers(r_max, B_s)
    if args.space_scale is None:
        space_scale = (r_max / max(B_s - 1, 1)) * 1.25
    else:
        space_scale = float(args.space_scale)
    space_scale = jnp.asarray(space_scale)

    # --- Optional "true" params for reporting
    mu_true, K_true, omega_true, sigma_true = reconstruct_true_params_if_present(
        np.asarray(params_init) if params_init is not None else None, N, M
    )
    M_K_true = np.asarray(data["mark_kernel_matrix"]) if "mark_kernel_matrix" in data else None

    # --- Inference
    if args.method == "mcmc":
        kernel = NUTS(hawkes_np_model, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                    num_chains=args.chains, chain_method="sequential")
        mcmc.run(
            key,
            t_obs=t, u_obs=u, e_obs=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            space_centers=space_centers, space_scale=space_scale,
            N=N, M=M
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()

        # Posterior means of everything we need
        mu_hat   = jnp.mean(posterior["mu"], axis=0)
        K_hat    = jnp.mean(posterior["K_masked"], axis=0)
        M_K_hat  = jnp.mean(posterior["M_K"], axis=0)
        # For kernels, keep draws of the weights to reconstruct shapes later
        a_draws  = posterior["a_uncon"]
        b_draws  = posterior["b_uncon"]

        # Save full posterior state for later plotting/diagnostics
        np.savez(
            "mcmc_state_np.npz",
            mu=np.asarray(posterior["mu"]),
            K_masked=np.asarray(posterior["K_masked"]),
            M_K=np.asarray(posterior["M_K"]),
            a_uncon=np.asarray(a_draws),
            b_uncon=np.asarray(b_draws),
            time_centers=np.asarray(time_centers),
            time_scale=float(time_scale),
            space_centers=np.asarray(space_centers),
            space_scale=float(space_scale),
            t=np.asarray(t), u=np.asarray(u), e=np.asarray(e), T=float(T),
            node_locations=np.asarray(node_locations),
            reach_mask=np.asarray(reach_mask_np),
            mu_true=np.asarray(mu_true) if mu_true is not None else None,
            K_true=np.asarray(K_true) if K_true is not None else None,
            sigma_true=float(sigma_true) if sigma_true is not None else None,
            omega_true=float(omega_true) if omega_true is not None else None,
            M_K_true=np.asarray(M_K_true) if M_K_true is not None else None
        )
        print("Saved full MCMC posterior to mcmc_state_np.npz")

    else:  # MAP (AutoDelta)
        guide = autoguide.AutoDelta(hawkes_np_model)
        svi = SVI(hawkes_np_model, guide, numpyro.optim.Adam(5e-2), loss=Trace_ELBO())
        state = svi.init(
            key,
            t_obs=t, u_obs=u, e_obs=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            time_centers=time_centers, time_scale=time_scale,
            space_centers=space_centers, space_scale=space_scale,
            N=N, M=M
        )
        for i in range(2000):
            state, loss = svi.update(
                state,
                t_obs=t, u_obs=u, e_obs=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                time_centers=time_centers, time_scale=time_scale,
                space_centers=space_centers, space_scale=space_scale,
                N=N, M=M
            )
            if (i + 1) % 200 == 0:
                print(f"[SVI] iter {i+1:04d} loss={float(loss):.3f}")
        params_map = svi.get_params(state)
        mu_hat  = params_map["mu"]
        K_hat   = params_map["K_masked"]
        M_K_hat = params_map["M_K"]

    # --- Report & save posterior means
    print("\n=== Posterior means (nonparam kernels) ===")
    print(f"mu_hat shape:  {tuple(np.asarray(mu_hat).shape)}")
    print(f"K_hat shape:   {tuple(np.asarray(K_hat).shape)}")
    print(f"M_K_hat shape: {tuple(np.asarray(M_K_hat).shape)}")

    out = {
        "mu_hat":    np.asarray(mu_hat),
        "K_hat":     np.asarray(K_hat),
        "M_K_hat":   np.asarray(M_K_hat),
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
    }
    if "mcmc" in args.method:
        out["mcmc_state_file"] = "mcmc_state_np.npz"

    with open("inference_result_np.pickle", "wb") as f:
        pickle.dump(out, f)
    print("\nSaved posterior means to inference_result_np.pickle")

    # Optional quick comparison if truths are present (sigma/omega won't match here
    # because kernels are nonparametric; but K/M_K can be compared)
    if mu_true is not None:
        rmse_mu = np.sqrt(np.mean((out["mu_hat"] - mu_true) ** 2))
        print(f"mu    RMSE vs true: {rmse_mu:.6f}")
    if "K_true" in data:
        rmse_K = np.sqrt(np.mean((out["K_hat"] - np.asarray(K_true)) ** 2))
        print(f"K     RMSE vs true: {rmse_K:.6f}")
    if "mark_kernel_matrix" in data:
        rmse_MK = np.sqrt(np.mean((out["M_K_hat"] - np.asarray(M_K_true)) ** 2))
        print(f"M_K   RMSE vs true: {rmse_MK:.6f}")


if __name__ == "__main__":
    main()
