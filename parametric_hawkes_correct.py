#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hawkes inference with NumPyro + JAX, directly consuming the pickle
produced by hawkes_generate.py.

Learns:
  - mu:       (N, M)
  - K_masked: (N, N)
  - sigma:    scalar
  - omega:    scalar
  - M_K:      (M, M)

Adds:
  - Informative priors centered at generator params (optional).
  - Prints true vs inferred (σ, ω, K, M_K, μ) with RMSE/rel. errors.
  - Saves full MCMC posterior to mcmc_state.npz for later reload.
"""

import argparse
import pickle
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
from numpyro import enable_x64

# Platform configuration (CPU-safe)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(10)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def compute_reachability(adjacency, num_hops=1):
    A = (adjacency > 0).astype(np.int32)
    N = A.shape[0]
    R = np.eye(N, dtype=np.int32)
    cur = A.copy()
    for _ in range(num_hops):
        R = (R | (cur > 0).astype(np.int32)).astype(np.int32)
        cur = (cur @ A > 0).astype(np.int32)
    return R.astype(np.float32)

def pairwise_sq_dists(node_xy):  # (N,2) -> (N,N)
    diff = node_xy[:, None, :] - node_xy[None, :, :]
    return jnp.sum(diff * diff, axis=-1)

def gaussian_kernel(d2, sigma):  # (N,N), scalar -> (N,N)
    denom = 2.0 * (sigma ** 2)
    norm = 1.0 / (2.0 * jnp.pi * (sigma ** 2))
    return norm * jnp.exp(-d2 / denom)

def prep_events_structured(events, num_event_types=None):
    t = np.asarray(events['t'])
    u = np.asarray(events['u'])
    e = np.asarray(events['e'])
    T = float(t.max()) if t.size > 0 else 0.0
    N = int(u.max()) + 1 if u.size > 0 else 0
    if num_event_types is None:
        M = int(e.max()) + 1 if e.size > 0 else 1
    else:
        M = int(num_event_types)
    return t, u, e, T, N, M

def reconstruct_true_params_if_present(params_init, N, M):
    """
    params_init: [mu.flatten(), K.flatten(), omega, sigma]
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

def inv_softplus(y, eps=1e-6):
    # inverse of softplus: x = log(exp(y)-1) , stabilized
    y = jnp.maximum(y, eps)
    return jnp.log(jnp.expm1(y))

def rmse(a, b):
    return float(jnp.sqrt(jnp.mean((jnp.asarray(a) - jnp.asarray(b))**2)))

def rel_fro_error(a, b, eps=1e-12):
    a = jnp.asarray(a); b = jnp.asarray(b)
    num = jnp.linalg.norm(a - b)
    den = jnp.linalg.norm(b) + eps
    return float(num / den)

# -----------------------------------------------------------------------------
# NumPyro model (with optional informative priors)
# -----------------------------------------------------------------------------

def hawkes_numpyro_model(t_obs, u_obs, e_obs, T, node_xy, reach_mask, N: int, M: int,
                         use_info_priors: bool,
                         mu0_uncon=None, K0_uncon=None, M0_uncon=None,
                         log_sigma0=None, log_omega0=None,
                         ps_mu=0.5, ps_K=0.3, ps_M=0.3, ps_logsig=0.25, ps_logom=0.25):
    """
    NumPyro model for multivariate marked Hawkes.

    If use_info_priors=True and *_0 args provided, centers priors at those values.
    ps_* are prior std devs.
    """
    # Priors for mu (positivity via softplus)
    if use_info_priors and (mu0_uncon is not None):
        mu_uncon = numpyro.sample("mu_uncon",
                                  dist.Normal(mu0_uncon, ps_mu).to_event(2))
    else:
        mu_uncon = numpyro.sample("mu_uncon",
                                  dist.Normal(0.0, 1.0).expand([N, M]).to_event(2))
    mu = numpyro.deterministic("mu", jax.nn.softplus(mu_uncon) + 1e-6)  # (N,M)

    # Priors for K (masked later, positivity via softplus)
    if use_info_priors and (K0_uncon is not None):
        K_uncon = numpyro.sample("K_uncon",
                                 dist.Normal(K0_uncon, ps_K).to_event(2))
    else:
        K_uncon = numpyro.sample("K_uncon",
                                 dist.Normal(0.0, 0.5).expand([N, N]).to_event(2))
    K_pos = jax.nn.softplus(K_uncon)
    K_masked = numpyro.deterministic("K_masked", K_pos * reach_mask)     # (N,N)

    # σ prior (log-scale)
    if use_info_priors and (log_sigma0 is not None):
        log_sigma = numpyro.sample("log_sigma", dist.Normal(log_sigma0, ps_logsig))
    else:
        log_sigma = numpyro.sample("log_sigma", dist.Normal(0.0, 0.5))
    sigma = numpyro.deterministic("sigma", jnp.exp(log_sigma) + 1e-6)

    # ω prior (log-scale)
    if use_info_priors and (log_omega0 is not None):
        log_omega = numpyro.sample("log_omega", dist.Normal(log_omega0, ps_logom))
    else:
        log_omega = numpyro.sample("log_omega", dist.Normal(0.0, 0.5))
    omega = numpyro.deterministic("omega", jnp.exp(log_omega) + 1e-6)

    # M_K prior (positivity via softplus)
    if use_info_priors and (M0_uncon is not None):
        M_uncon = numpyro.sample("M_uncon", dist.Normal(M0_uncon, ps_M).to_event(2))
    else:
        M_uncon = numpyro.sample("M_uncon", dist.Normal(0.0, 0.5).expand([M, M]).to_event(2))
    M_K = numpyro.deterministic("M_K", jax.nn.softplus(M_uncon) + 1e-6)  # (M,M)

    # Spatial kernel and effective node kernel
    d2 = pairwise_sq_dists(node_xy)                      # (N,N)
    kappa = numpyro.deterministic("kappa", gaussian_kernel(d2, sigma))  # (N,N)
    G = numpyro.deterministic("G", K_masked * kappa)                     # (N,N)

    # Sort events by time
    order = jnp.argsort(t_obs)
    t = t_obs[order]
    u = u_obs[order]
    e = e_obs[order]

    # State S stores S_{v,n}(t) for all (v,n), shape (N,M)
    S0 = jnp.zeros((N, M), dtype=t_obs.dtype)

    def step(carry, inputs):
        S_prev, t_prev, loglik_sum = carry
        t_i, u_i, e_i = inputs
        dt = t_i - t_prev
        decay = jnp.exp(-omega * dt)
        S = S_prev * decay

        # intensity matrix at t_i^-: lam_mat = mu + (G @ S) @ M_K^T
        GS = G @ S                              # (N,M)
        lam_mat = mu + GS @ M_K.T               # (N,M)
        lam_ie = jnp.clip(lam_mat[u_i, e_i], a_min=1e-12)
        loglik_sum = loglik_sum + jnp.log(lam_ie)

        # jump: S_{u_i,e_i} += omega
        S = S.at[u_i, e_i].add(omega)
        return (S, t_i, loglik_sum), None

    (S_T, t_last, event_loglik), _ = lax.scan(
        step,
        (S0, jnp.array(0.0, dtype=t_obs.dtype), jnp.array(0.0)),
        (t, u, e),
        length=t.shape[0]
    )

    # Integral terms
    base_int = T * jnp.sum(mu)

    one_minus_tail = 1.0 - jnp.exp(-omega * (T - t))    # (K,)
    S_int = jnp.zeros((N, M), dtype=t.dtype)
    S_int = S_int.at[u, e].add(one_minus_tail)          # (N,M)

    colsum_G = jnp.sum(G, axis=0)       # (N,)
    colsum_MK = jnp.sum(M_K, axis=0)    # (M,)

    exc_int = jnp.einsum('vn,v,n->', S_int, colsum_G, colsum_MK)
    loglik = event_loglik - base_int - exc_int
    numpyro.factor("loglik", loglik)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NumPyro inference for networked marked Hawkes from generator pickle.")
    parser.add_argument("--data", type=str, default="traffic_hawkes_simulation2.pickle",
                        help="Path to pickle produced by hawkes_generate.py")
    parser.add_argument("--method", type=str, choices=["mcmc", "map"], default="mcmc",
                        help="Inference method: mcmc (NUTS) or map (AutoDelta)")
    parser.add_argument("--warmup", type=int, default=5000, help="MCMC warmup steps")
    parser.add_argument("--samples", type=int, default=5000, help="MCMC samples per chain")
    parser.add_argument("--chains", type=int, default=8, help="Number of chains")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--x64", action="store_true", help="Enable float64 (recommended)")
    parser.add_argument("--use_info_priors", dest="use_info_priors", action="store_true", default=True,
                        help="Center priors at generator values if available.")
    parser.add_argument("--no_use_info_priors", dest="use_info_priors", action="store_false")
    # Prior std hyperparams
    parser.add_argument("--ps_mu", type=float, default=0.5)
    parser.add_argument("--ps_K", type=float, default=0.3)
    parser.add_argument("--ps_M", type=float, default=0.3)
    parser.add_argument("--ps_logsig", type=float, default=0.25)
    parser.add_argument("--ps_logom", type=float, default=0.25)

    args = parser.parse_args()
    if args.x64:
        enable_x64()

    # ------------------ Load generator pickle ------------------
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    events = data["events"]
    num_nodes = int(data["num_nodes"])
    num_event_types = int(data["num_event_types"])
    node_locations = np.asarray(data["node_locations"], dtype=float)  # (N,2)
    adjacency = np.asarray(data["adjacency_matrix"], dtype=float)     # (N,N)
    num_hops = int(data.get("num_hops", 1))
    params_init = data.get("params", None)   # [mu.flatten(), K.flatten(), omega, sigma]
    M_K_true_np = data.get("mark_kernel_matrix", None)

    # ------------------ Prepare arrays ------------------
    t_np, u_np, e_np, T_np, N_from_ev, M_from_ev = prep_events_structured(events, num_event_types)
    assert num_nodes == N_from_ev, "Mismatch N from pickle vs events."
    assert num_event_types == M_from_ev, "Mismatch M from pickle vs events."

    reach_mask_np = compute_reachability(adjacency, num_hops=num_hops)

    # JAX arrays
    key = jax.random.PRNGKey(args.seed)
    t = jnp.asarray(t_np)
    u = jnp.asarray(u_np)
    e = jnp.asarray(e_np)
    T = jnp.asarray(T_np, dtype=t.dtype)
    node_xy = jnp.asarray(node_locations)
    reach_mask = jnp.asarray(reach_mask_np)

    N = int(num_nodes)
    M = int(num_event_types)

    # Optional: recover "true" params for informative priors and reporting
    mu_true, K_true, omega_true, sigma_true = reconstruct_true_params_if_present(
        np.asarray(params_init) if params_init is not None else None, N, M
    )
    use_info_priors = bool(args.use_info_priors and (mu_true is not None) and (K_true is not None)
                           and (omega_true is not None) and (sigma_true is not None)
                           and (M_K_true_np is not None))

    # Build prior centers on unconstrained spaces if available
    mu0_uncon = None; K0_uncon = None; M0_uncon = None
    log_sigma0 = None; log_omega0 = None
    if use_info_priors:
        eps = 1e-4
        mu0_uncon = inv_softplus(jnp.asarray(mu_true) + eps)
        K0_uncon  = inv_softplus(jnp.asarray(K_true) + eps)
        M0_uncon  = inv_softplus(jnp.asarray(M_K_true_np) + eps)
        log_sigma0 = jnp.log(jnp.asarray(sigma_true))
        log_omega0 = jnp.log(jnp.asarray(omega_true))

    # ------------------ Inference ------------------
    if args.method == "mcmc":
        kernel = NUTS(hawkes_numpyro_model, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                    num_chains=args.chains, chain_method="parallel")
        mcmc.run(
            key,
            t_obs=t, u_obs=u, e_obs=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            N=N, M=M,
            use_info_priors=use_info_priors,
            mu0_uncon=mu0_uncon, K0_uncon=K0_uncon, M0_uncon=M0_uncon,
            log_sigma0=log_sigma0, log_omega0=log_omega0,
            ps_mu=args.ps_mu, ps_K=args.ps_K, ps_M=args.ps_M,
            ps_logsig=args.ps_logsig, ps_logom=args.ps_logom
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()
        posterior_by_chain = mcmc.get_samples(group_by_chain=True)

        # Posterior means
        mu_hat = jnp.mean(posterior["mu"], axis=0)
        K_hat = jnp.mean(posterior["K_masked"], axis=0)
        sigma_hat = jnp.mean(posterior["sigma"])
        omega_hat = jnp.mean(posterior["omega"])
        M_K_hat = jnp.mean(posterior["M_K"], axis=0)

        # Save MCMC state for later reload
        # (np.savez works well for arrays; we store true params too for context)
        np.savez(
            "mcmc_state.npz",
            mu=np.asarray(posterior_by_chain["mu"]),
            K_masked=np.asarray(posterior_by_chain["K_masked"]),
            sigma=np.asarray(posterior_by_chain["sigma"]),
            omega=np.asarray(posterior_by_chain["omega"]),
            M_K=np.asarray(posterior_by_chain["M_K"]),
            t=t_np, u=u_np, e=e_np, T=T_np,
            node_locations=node_locations, reach_mask=reach_mask_np,
            mu_true=np.asarray(mu_true) if mu_true is not None else np.array([]),
            K_true=np.asarray(K_true) if K_true is not None else np.array([]),
            sigma_true=np.array([sigma_true]) if sigma_true is not None else np.array([]),
            omega_true=np.array([omega_true]) if omega_true is not None else np.array([]),
            M_K_true=np.asarray(M_K_true_np) if M_K_true_np is not None else np.array([])
        )
        print("\nSaved full MCMC state to mcmc_state.npz")

    else:  # MAP via AutoDelta
        guide = autoguide.AutoDelta(hawkes_numpyro_model)
        svi = SVI(hawkes_numpyro_model, guide, numpyro.optim.Adam(5e-2), loss=Trace_ELBO())
        state = svi.init(
            key,
            t_obs=t, u_obs=u, e_obs=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
            N=N, M=M,
            use_info_priors=use_info_priors,
            mu0_uncon=mu0_uncon, K0_uncon=K0_uncon, M0_uncon=M0_uncon,
            log_sigma0=log_sigma0, log_omega0=log_omega0,
            ps_mu=args.ps_mu, ps_K=args.ps_K, ps_M=args.ps_M,
            ps_logsig=args.ps_logsig, ps_logom=args.ps_logom
        )
        for i in range(3000):
            state, loss = svi.update(
                state,
                t_obs=t, u_obs=u, e_obs=e, T=T, node_xy=node_xy, reach_mask=reach_mask,
                N=N, M=M,
                use_info_priors=use_info_priors,
                mu0_uncon=mu0_uncon, K0_uncon=K0_uncon, M0_uncon=M0_uncon,
                log_sigma0=log_sigma0, log_omega0=log_omega0,
                ps_mu=args.ps_mu, ps_K=args.ps_K, ps_M=args.ps_M,
                ps_logsig=args.ps_logsig, ps_logom=args.ps_logom
            )
            if (i+1) % 300 == 0:
                print(f"[SVI] iter {i+1:04d} loss={float(loss):.3f}")
        params_map = svi.get_params(state)
        mu_hat = params_map["mu"]
        K_hat = params_map["K_masked"]
        sigma_hat = params_map["sigma"]
        omega_hat = params_map["omega"]
        M_K_hat = params_map["M_K"]
        posterior = None  # not saved for MAP

    # ------------------ Report & Save ------------------
    print("\n=== Posterior (or MAP) means ===")
    print(f"mu_hat shape: {tuple(mu_hat.shape)}")
    print(f"K_hat shape:  {tuple(K_hat.shape)}")
    print(f"M_K_hat shape:{tuple(M_K_hat.shape)}")
    print(f"sigma_hat:    {float(sigma_hat):.6f}")
    print(f"omega_hat:    {float(omega_hat):.6f}")

    # True vs inferred comparison
    if (mu_true is not None) and (K_true is not None) and (sigma_true is not None) and (omega_true is not None):
        print("\n--- True vs Inferred ---")
        print(f"sigma: true={sigma_true:.6f}  inferred={float(sigma_hat):.6f}  "
              f"rel.err={abs(float(sigma_hat)-sigma_true)/max(sigma_true,1e-12):.3%}")
        print(f"omega: true={omega_true:.6f}  inferred={float(omega_hat):.6f}  "
              f"rel.err={abs(float(omega_hat)-omega_true)/max(omega_true,1e-12):.3%}")
        k_rmse = rmse(K_hat, K_true)
        k_rel = rel_fro_error(K_hat, K_true)
        print(f"K:     RMSE={k_rmse:.6f}  rel.Fro.err={k_rel:.3%}")
        mu_rmse = rmse(mu_hat, mu_true)
        print(f"mu:    RMSE={mu_rmse:.6f}")
    if M_K_true_np is not None:
        mk_rmse = rmse(M_K_hat, M_K_true_np)
        mk_rel = rel_fro_error(M_K_hat, M_K_true_np)
        print(f"M_K:   RMSE={mk_rmse:.6f}  rel.Fro.err={mk_rel:.3%}")

    # Save summary pickle (posterior means + truths)
    out = {
        "mu_hat":    np.asarray(mu_hat),
        "K_hat":     np.asarray(K_hat),
        "M_K_hat":   np.asarray(M_K_hat),
        "sigma_hat": float(sigma_hat),
        "omega_hat": float(omega_hat),
        "N": N,
        "M": M,
        "T": float(T),
        "node_locations": np.asarray(node_locations),
        "reach_mask": np.asarray(reach_mask_np),
        "data_pickle": args.data,
        "method": args.method,
        "used_informative_priors": use_info_priors,
        "prior_stds": dict(ps_mu=args.ps_mu, ps_K=args.ps_K, ps_M=args.ps_M,
                           ps_logsig=args.ps_logsig, ps_logom=args.ps_logom),
    }
    if mu_true is not None:
        out["mu_true"] = np.asarray(mu_true)
        out["K_true"] = np.asarray(K_true)
        out["sigma_true"] = float(sigma_true)
        out["omega_true"] = float(omega_true)
    if M_K_true_np is not None:
        out["M_K_true"] = np.asarray(M_K_true_np)

    with open("inference_result.pickle", "wb") as f:
        pickle.dump(out, f)
    print("\nSaved posterior means to inference_result.pickle")
    if args.method == "mcmc":
        print("Saved full MCMC posterior to mcmc_state.npz")

if __name__ == "__main__":
    main()
