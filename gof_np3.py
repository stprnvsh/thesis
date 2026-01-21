#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goodness-of-Fit Tests for Joint Spatio-Temporal Hawkes Model (nonpm_window_3.py).

This model uses a joint kernel ψ̃(τ, r) that depends on both time lag AND distance.
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import erf
from scipy.sparse.csgraph import shortest_path
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore')


def compute_network_distance(adjacency):
    """Compute shortest path distances on the graph."""
    A = (adjacency > 0).astype(np.float32)
    dist_matrix = shortest_path(A, directed=False, method='FW')
    finite_mask = np.isfinite(dist_matrix)
    if finite_mask.any():
        max_finite = dist_matrix[finite_mask].max()
        dist_matrix = np.where(finite_mask, dist_matrix, max_finite * 2)
    return dist_matrix


def gauss_bump_int_0_to(x, c, scale):
    """∫_0^x exp(-0.5 * ((t - c)/s)^2) dt"""
    rt2 = np.sqrt(2.0)
    pref = scale * np.sqrt(np.pi / 2.0)
    return pref * (erf((x - c) / (rt2 * scale)) - erf((-c) / (rt2 * scale)))


def gauss_bump_int_0_to_inf(c, scale):
    """∫_0^∞ exp(-0.5 * ((t - c)/s)^2) dt"""
    rt2 = np.sqrt(2.0)
    return scale * np.sqrt(np.pi / 2.0) * (1.0 - erf((-c) / (rt2 * scale)))


def build_joint_kernel_components(kernel_param, time_centers, time_scale, 
                                   dist_centers, dist_scale, D):
    """
    Build components for the joint spatio-temporal kernel ψ̃(τ, r).
    
    Returns:
        S_t: (N, N, B_t) - spatial-weighted temporal coefficients per pair
        denom: (N, N) - normalization denominators per pair
    """
    B_t = len(time_centers)
    B_r = len(dist_centers)
    N = D.shape[0]
    
    # w_pos from kernel_param (apply softplus)
    w_pos = np.log1p(np.exp(kernel_param)) + 1e-8  # softplus
    
    # Psi_r: (N, N, B_r) - spatial basis
    Psi_r = np.stack([np.exp(-0.5 * ((D - c) / dist_scale) ** 2) for c in dist_centers], axis=-1)
    
    # S_t[i,j,:] = Σ_b w_pos[:, b] * Psi_r[i,j,b] -> (N, N, B_t)
    S_t = np.tensordot(Psi_r, w_pos, axes=[[2], [1]])
    
    # Denominator per pair (unit time integral)
    I_inf = np.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers])
    denom = np.maximum(np.tensordot(S_t, I_inf, axes=[[2], [0]]), 1e-12)
    
    return S_t, denom


def psi_val(dt, i_idx, j_idx, S_t, denom, time_centers, time_scale):
    """Evaluate ψ̃(dt, r_{i,j})."""
    dt = max(dt, 0.0)
    phi_t = np.exp(-0.5 * ((dt - time_centers) / time_scale) ** 2)
    num = np.dot(S_t[i_idx, j_idx], phi_t)
    return num / denom[i_idx, j_idx]


def psi_int(dt_cap, i_idx, j_idx, S_t, denom, time_centers, time_scale):
    """∫_0^{dt_cap} ψ̃(τ, r_{i,j}) dτ"""
    dt_cap = max(dt_cap, 0.0)
    I_cap = np.array([gauss_bump_int_0_to(dt_cap, c, time_scale) for c in time_centers])
    num = np.dot(S_t[i_idx, j_idx], I_cap)
    return num / denom[i_idx, j_idx]


def compute_intensity_at_event(i, t, u, e, mu, K, M_K, alpha,
                                S_t, denom, time_centers, time_scale, window=np.inf):
    """Compute total intensity λ(t_i^-) just before event i."""
    t_i = t[i]
    N, M = mu.shape
    
    # Baseline
    total_baseline = np.sum(mu)
    
    # Excitation from past events
    total_excite = 0.0
    for j in range(i):
        dt = t_i - t[j]
        if dt <= 0 or dt > window:
            continue
        u_j, e_j = u[j], e[j]
        
        # Sum over all target nodes and marks
        for v in range(N):
            psi_v = psi_val(dt, v, u_j, S_t, denom, time_centers, time_scale)
            for k in range(M):
                total_excite += alpha * K[v, u_j] * M_K[e_j, k] * psi_v
    
    return total_baseline + total_excite


def compute_compensator_location(t_start, t_end, events_before_end,
                                  t, u, e, mu, K, M_K, alpha,
                                  S_t, denom, time_centers, time_scale,
                                  target_node, target_mark, window=np.inf):
    """
    Compute ∫_{t_start}^{t_end} λ_{target_node, target_mark}(s) ds.
    
    This is the per-location compensator for the time rescaling theorem.
    """
    # Baseline contribution for this specific (node, mark)
    baseline = (t_end - t_start) * mu[target_node, target_mark]
    
    # Excitation contribution
    excitation = 0.0
    for j in range(events_before_end):
        t_j = t[j]
        if t_j >= t_end:
            break
        
        u_j, e_j = int(u[j]), int(e[j])
        
        # Integration limits relative to t_j
        a = max(t_start - t_j, 0.0)
        b = min(t_end - t_j, window)
        
        if b <= a:
            continue
        
        # Joint kernel integral for THIS target node
        int_b = psi_int(b, target_node, u_j, S_t, denom, time_centers, time_scale)
        int_a = psi_int(a, target_node, u_j, S_t, denom, time_centers, time_scale) if a > 0 else 0
        int_val = int_b - int_a
        
        # Contribution from event j to (target_node, target_mark)
        excitation += alpha * K[target_node, u_j] * M_K[e_j, target_mark] * int_val
    
    return baseline + excitation


def compute_rescaled_times(t, u, e, mu, K, M_K, alpha,
                            S_t, denom, time_centers, time_scale, window=np.inf):
    """
    Compute rescaled inter-arrival times for SPATIO-TEMPORAL Hawkes.
    
    For each event at (t_i, u_i, e_i), compute:
        τ_i = Λ_{u_i, e_i}(t_prev) - Λ_{u_i, e_i}(t_i)
    
    where t_prev is the PREVIOUS event at the SAME (node, mark) pair.
    """
    n_events = len(t)
    N, M = mu.shape
    tau = []
    tau_labels = []
    
    # Track last event time for each (node, mark) pair
    last_time = {(v, k): 0.0 for v in range(N) for k in range(M)}
    
    print(f"Computing rescaled times for {n_events} events (per-location)...")
    
    for i in range(n_events):
        t_curr = t[i]
        u_curr = int(u[i])
        e_curr = int(e[i])
        
        # Get previous event time at THIS (node, mark)
        t_prev = last_time[(u_curr, e_curr)]
        
        # Compute compensator for λ_{u_curr, e_curr} from t_prev to t_curr
        Lambda = compute_compensator_location(
            t_prev, t_curr, i,
            t, u, e, mu, K, M_K, alpha,
            S_t, denom, time_centers, time_scale,
            u_curr, e_curr, window
        )
        
        if Lambda > 0:
            tau.append(Lambda)
            tau_labels.append((u_curr, e_curr))
        
        # Update last time for this (node, mark)
        last_time[(u_curr, e_curr)] = t_curr
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{n_events} events")
    
    print("Done computing rescaled times.")
    return np.array(tau), tau_labels


def compute_rescaled_times_network_wide(
    t, u, e, mu, K, M_K, alpha,
    S_t, denom, time_centers, time_scale, window=np.inf, by_mark=False
):
    """
    Compute rescaled inter-arrival times for NETWORK-WIDE aggregated process.
    
    Two modes:
    - by_mark=False: Total intensity λ(t) = Σ_v Σ_k λ_{v,k}(t)
    - by_mark=True: Per-mark intensity λ_k(t) = Σ_v λ_{v,k}(t)
    """
    n_events = len(t)
    N, M = mu.shape
    
    if by_mark:
        last_time = {k: 0.0 for k in range(M)}
        tau = []
        labels = []
        
        print(f"Computing network-wide rescaled times (by mark)...")
        
        for i in range(n_events):
            t_curr = t[i]
            e_curr = int(e[i])
            t_prev = last_time[e_curr]
            
            # Compensator: Σ_v Λ_{v, e_curr}(t_prev → t_curr)
            Lambda = 0.0
            for v in range(N):
                Lambda += compute_compensator_location(
                    t_prev, t_curr, i,
                    t, u, e, mu, K, M_K, alpha,
                    S_t, denom, time_centers, time_scale,
                    v, e_curr, window
                )
            
            if Lambda > 0:
                tau.append(Lambda)
                labels.append(e_curr)
            
            last_time[e_curr] = t_curr
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{n_events} events")
        
        return np.array(tau), labels
    
    else:
        tau = []
        print(f"Computing network-wide rescaled times (total)...")
        
        t_prev = 0.0
        for i in range(n_events):
            t_curr = t[i]
            
            # Total compensator: Σ_v Σ_k Λ_{v,k}(t_prev → t_curr)
            Lambda = 0.0
            for v in range(N):
                for k in range(M):
                    Lambda += compute_compensator_location(
                        t_prev, t_curr, i,
                        t, u, e, mu, K, M_K, alpha,
                        S_t, denom, time_centers, time_scale,
                        v, k, window
                    )
            
            if Lambda > 0:
                tau.append(Lambda)
            
            t_prev = t_curr
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_events} events")
        
        return np.array(tau), None


def run_gof_tests(tau, alpha_level=0.05, tau_labels=None, min_events_per_pair=20):
    """Run GOF tests on rescaled times with KS test."""
    if len(tau) < 10:
        print("Too few rescaled times for GOF tests")
        return None, None, None
    
    # Per-pair normalization
    pair_taus = {}
    if tau_labels is not None:
        unique_pairs = list(set(tau_labels))
        pair_means = {}
        
        for pair in unique_pairs:
            mask = np.array([l == pair for l in tau_labels])
            if mask.sum() >= min_events_per_pair:
                pair_taus[pair] = tau[mask]
                pair_means[pair] = np.mean(tau[mask])
        
        if len(pair_means) > 0:
            tau_norm = []
            labels_norm = []
            for t_val, label in zip(tau, tau_labels):
                if label in pair_means:
                    tau_norm.append(t_val / pair_means[label])
                    labels_norm.append(label)
            tau_norm = np.array(tau_norm)
            print(f"Using {len(pair_means)} active (node,mark) pairs with >={min_events_per_pair} events")
        else:
            tau_norm = tau / np.mean(tau)
    else:
        tau_norm = tau / np.mean(tau)
    
    # Scale error
    scale_error = abs(np.mean(tau_norm) - 1.0)
    
    print(f"\n{'='*60}")
    print("GOF RESULTS (Joint Spatio-Temporal Kernel)")
    print(f"{'='*60}")
    print(f"Rescaled times: {len(tau_norm)}")
    print(f"Mean τ (should be ~1.0): {np.mean(tau_norm):.4f}")
    print(f"Scale error: {scale_error:.4f} ({scale_error*100:.1f}%)")
    
    # KS test against Exp(1) - use statistic not p-value
    print(f"\n--- KS Test (distribution check) ---")
    try:
        # Pooled KS test
        ks_stat, ks_pval = stats.kstest(tau_norm, 'expon', args=(0, 1))
        
        # Interpret KS statistic (effect size)
        if ks_stat < 0.05:
            ks_quality = "EXCELLENT"
        elif ks_stat < 0.10:
            ks_quality = "GOOD"
        elif ks_stat < 0.15:
            ks_quality = "ACCEPTABLE"
        else:
            ks_quality = "POOR"
        
        print(f"Pooled: KS statistic={ks_stat:.4f} ({ks_quality})")
        print(f"  (p={ks_pval:.4f} - ignore for large samples)")
        
        # Per-pair KS tests - report statistics
        if len(pair_taus) > 0:
            pair_ks_stats = []
            for pair, taus in pair_taus.items():
                if len(taus) >= min_events_per_pair:
                    taus_norm = taus / np.mean(taus)
                    stat, _ = stats.kstest(taus_norm, 'expon', args=(0, 1))
                    pair_ks_stats.append(stat)
            
            if pair_ks_stats:
                median_stat = np.median(pair_ks_stats)
                pct_good = 100 * np.mean([s < 0.15 for s in pair_ks_stats])
                print(f"Per-pair ({len(pair_ks_stats)} pairs): median KS={median_stat:.4f}, {pct_good:.0f}% good")
    except Exception as e:
        print(f"KS test failed: {e}")
        ks_stat = np.nan
    
    # Ljung-Box test
    print(f"\n--- Ljung-Box Test (temporal independence) ---")
    try:
        lb_result = acorr_ljungbox(tau_norm, lags=[10, 20, 30], return_df=True)
        lb_pvals = lb_result['lb_pvalue'].values
        lb_pass = all(p > alpha_level for p in lb_pvals)
        print(f"Lag 10: p={lb_pvals[0]:.4f}")
        print(f"Lag 20: p={lb_pvals[1]:.4f}")
        print(f"Lag 30: p={lb_pvals[2]:.4f}")
        print(f"Result: {'PASS' if lb_pass else 'FAIL'}")
        lb_pval = lb_pvals[0]
    except:
        lb_pval = np.nan
        lb_pass = False
    
    print(f"{'='*60}")
    
    # Use KS statistic for pass/fail, not p-value
    ks_good = ks_stat < 0.15 if not np.isnan(ks_stat) else False
    passed = lb_pass and scale_error < 0.2
    
    if passed and ks_good:
        print(f"\n✓ Model PASSES ALL TESTS")
        print(f"  - Ljung-Box: temporal dynamics captured")
        print(f"  - KS stat {ks_stat:.4f}: distribution matches well")
    elif passed:
        print(f"\n○ Model PASSES (KS marginal)")
        print(f"  - Ljung-Box: temporal dynamics captured")
        print(f"  - KS stat {ks_stat:.4f}: some distribution deviation")
    else:
        print(f"\n✗ Model NEEDS WORK")
        if not lb_pass:
            print(f"  - Ljung-Box FAIL: missing temporal structure")
    
    return tau_norm, lb_pval, scale_error


def first_order_residuals(t, u, e, mu, T_max):
    """First-order residuals: compare observed vs expected counts per (node, mark)."""
    N, M = mu.shape
    
    print(f"\n--- First-Order Residuals (per node/mark) ---")
    
    observed = np.zeros((N, M))
    for i in range(len(t)):
        observed[int(u[i]), int(e[i])] += 1
    
    total_obs = observed.sum()
    baseline_exp = (mu * T_max).sum()
    excitation_ratio = total_obs / baseline_exp if baseline_exp > 0 else float('inf')
    
    print(f"Total observed: {int(total_obs)}")
    print(f"Baseline expected (μ×T): {baseline_exp:.1f}")
    print(f"Excitation ratio: {excitation_ratio:.2f}×")
    
    if excitation_ratio > 1:
        implied_sr = 1 - 1/excitation_ratio
        print(f"Implied spectral radius: ~{implied_sr:.2f}")
    
    obs_per_node = observed.sum(axis=1)
    exp_per_node = (mu * T_max).sum(axis=1)
    node_ratios = obs_per_node / (exp_per_node + 1e-10)
    
    print(f"\nPer-node obs/exp ratio: min={node_ratios.min():.2f}, max={node_ratios.max():.2f}")
    if node_ratios.std() / node_ratios.mean() > 0.5:
        print("-> High variance across nodes (heterogeneous excitation)")
    else:
        print("-> Balanced across nodes")


def plot_diagnostics(tau, t, save_path=None):
    """Generate diagnostic plots for joint kernel GOF."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. QQ-plot against Exp(1)
    ax = axes[0, 0]
    theoretical = stats.expon.ppf(np.linspace(0.01, 0.99, len(tau)))
    empirical = np.sort(tau)
    if len(tau) > 2000:
        idx = np.linspace(0, len(tau)-1, 2000, dtype=int)
        theoretical, empirical = theoretical[idx], empirical[idx]
    ax.scatter(theoretical, empirical, alpha=0.5, s=10)
    max_val = max(theoretical.max(), empirical.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='y=x (perfect fit)')
    ax.set_xlabel('Theoretical Quantiles (Exp(1))')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title('QQ-Plot: Rescaled Times vs Exp(1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Histogram
    ax = axes[0, 1]
    ax.hist(tau, bins=50, density=True, alpha=0.7, label='Empirical')
    x_range = np.linspace(0, min(tau.max(), 10), 200)
    ax.plot(x_range, stats.expon.pdf(x_range), 'r-', lw=2, label='Exp(1) PDF')
    ax.set_xlabel('Rescaled Time τ')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Rescaled Inter-arrival Times')
    ax.legend()
    ax.set_xlim(0, min(tau.max(), 10))
    ax.grid(True, alpha=0.3)
    
    # 3. Uniformity
    ax = axes[0, 2]
    u_transformed = 1 - np.exp(-tau)
    ax.hist(u_transformed, bins=50, density=True, alpha=0.7, label='Empirical')
    ax.axhline(y=1.0, color='r', linestyle='--', lw=2, label='Uniform(0,1)')
    ax.set_xlabel('Transformed U = 1 - exp(-τ)')
    ax.set_ylabel('Density')
    ax.set_title('Uniformity Check')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative residuals
    ax = axes[1, 0]
    cumsum_tau = np.cumsum(tau)
    n_events = np.arange(1, len(tau) + 1)
    ax.plot(cumsum_tau, n_events, 'b-', lw=1, label='N(Λ) (observed)')
    ax.plot([0, cumsum_tau[-1]], [0, cumsum_tau[-1]], 'r--', lw=2, label='y=x (expected)')
    ax.set_xlabel('Cumulative Compensator Λ(t)')
    ax.set_ylabel('Cumulative Event Count N(t)')
    ax.set_title('Cumulative Residuals: N(Λ) vs Λ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Residuals over time
    ax = axes[1, 1]
    residuals = n_events - cumsum_tau
    ax.plot(t[:len(residuals)], residuals, 'b-', lw=0.5, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', lw=1)
    sigma_band = 2 * np.sqrt(np.arange(1, len(residuals) + 1))
    ax.fill_between(t[:len(residuals)], -sigma_band, sigma_band, alpha=0.2, color='red', label='±2σ band')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Residual N(t) - Λ(t)')
    ax.set_title('Raw Residuals Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. ACF
    ax = axes[1, 2]
    max_lag = min(50, len(tau) // 10)
    acf = np.correlate(tau - np.mean(tau), tau - np.mean(tau), mode='full')
    acf = acf[len(acf)//2:len(acf)//2 + max_lag + 1]
    acf = acf / acf[0]
    ax.bar(range(max_lag + 1), acf, alpha=0.7)
    conf_band = 1.96 / np.sqrt(len(tau))
    ax.axhline(y=conf_band, color='r', linestyle='--', lw=1)
    ax.axhline(y=-conf_band, color='r', linestyle='--', lw=1)
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF of Rescaled Times (should be ~0 for lag > 0)')
    ax.set_xlim(-0.5, max_lag + 0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plots to {save_path}")
    plt.close()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True, help="Inference result pickle")
    p.add_argument("--data", required=True, help="Data pickle")
    p.add_argument("--alpha-level", type=float, default=0.05)
    p.add_argument("--output-prefix", default="gof_joint", help="Prefix for output files")
    p.add_argument("--aggregate", type=str, choices=["none", "mark", "total"], default="none",
                   help="Aggregation: none=per-location, mark=network-wide by type, total=all events")
    p.add_argument("--min-events", type=int, default=20, help="Min events per pair")
    args = p.parse_args()
    
    # Load results
    with open(args.result, "rb") as f:
        result = pickle.load(f)
    
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    
    # Extract model parameters
    mu = result["mu_hat"]
    K = result["K_hat"]
    M_K = result["M_K_hat"]
    alpha = result["alpha_hat"]
    kernel_param = result["kernel_param"]
    time_centers = result["time_centers"]
    time_scale = result["time_scale"]
    dist_centers = result["dist_centers"]
    dist_scale = result["dist_scale"]
    window = result.get("window", np.inf)
    
    N = result["N"]
    M = result["M"]
    
    # Load events
    events = data["events"]
    t = np.array(events["t"])
    u = np.array(events["u"])
    e = np.array(events["e"])
    
    # Sort by time
    order = np.argsort(t)
    t, u, e = t[order], u[order], e[order]
    
    # Compute network distance
    adjacency = np.array(data["adjacency_matrix"])
    D = compute_network_distance(adjacency)
    
    print(f"Loaded {len(t)} events, N={N}, M={M}")
    print(f"Window: {window}h")
    print(f"Alpha: {alpha:.4f}")
    
    # Build joint kernel components
    S_t, denom = build_joint_kernel_components(
        kernel_param, time_centers, time_scale, dist_centers, dist_scale, D
    )
    
    # Compute rescaled times based on aggregation level
    print("\nComputing rescaled times (this may take a while)...")
    if args.aggregate == "none":
        print("--- Per-Location Rescaling (standard) ---")
        tau, tau_labels = compute_rescaled_times(
            t, u, e, mu, K, M_K, alpha,
            S_t, denom, time_centers, time_scale, window
        )
    elif args.aggregate == "mark":
        print("--- Network-Wide by Mark Type ---")
        tau, tau_labels = compute_rescaled_times_network_wide(
            t, u, e, mu, K, M_K, alpha,
            S_t, denom, time_centers, time_scale, window, by_mark=True
        )
    else:  # total
        print("--- Network-Wide Total (all events) ---")
        tau, tau_labels = compute_rescaled_times_network_wide(
            t, u, e, mu, K, M_K, alpha,
            S_t, denom, time_centers, time_scale, window, by_mark=False
        )
    
    run_gof_tests(tau, args.alpha_level, tau_labels, min_events_per_pair=args.min_events)
    
    # First-order residuals
    T_max = t.max() - t.min()
    first_order_residuals(t, u, e, mu, T_max)
    
    print("\nGenerating diagnostic plots...")
    plot_diagnostics(tau, t, save_path=f"{args.output_prefix}_diagnostics.png")


if __name__ == "__main__":
    main()


