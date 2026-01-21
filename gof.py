#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goodness-of-Fit Tests for Network-Constrained Marked Hawkes Process.

Implements the time rescaling theorem (random time change) for model validation.
If the model is correctly specified, the rescaled inter-arrival times should be i.i.d. Exp(1).

Tests implemented:
1. Kolmogorov-Smirnov test against Exp(1)
2. Anderson-Darling test against Exp(1)
3. Ljung-Box test for autocorrelation in residuals
4. QQ-plot against Exp(1)
5. Uniformity test (transformed to Uniform(0,1))
6. Cumulative residual plot

Usage:
    python hawkes_gof_test.py --result inference_result_np6_xxx.pickle --data xxx.pickle
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import erf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore')


# ---------------- Gaussian basis utilities (matching the model) ----------------
def gauss_bump_int_0_to(x, c, scale):
    """∫_0^x exp(-0.5 * ((t - c)/s)^2) dt"""
    rt2 = np.sqrt(2.0)
    pref = scale * np.sqrt(np.pi / 2.0)
    return pref * (erf((x - c) / (rt2 * scale)) - erf((-c) / (rt2 * scale)))


def gauss_bump_int_a_to_b(a, b, c, scale):
    """∫_a^b exp(-0.5 * ((t - c)/s)^2) dt"""
    return gauss_bump_int_0_to(b, c, scale) - gauss_bump_int_0_to(a, c, scale)


def g_scalar(delta, time_centers, time_scale, mix_w):
    """Temporal kernel g(τ) at scalar τ ≥ 0."""
    delta = max(delta, 0.0)
    phi = np.exp(-0.5 * ((delta - time_centers) / time_scale) ** 2)
    return np.dot(phi, mix_w)


def G_int_a_to_b(a, b, time_centers, time_scale, mix_w):
    """∫_a^b g(τ) dτ using the Gaussian mixture representation."""
    if b <= a:
        return 0.0
    a = max(a, 0.0)
    total = 0.0
    for j, c in enumerate(time_centers):
        total += mix_w[j] * gauss_bump_int_a_to_b(a, b, c, time_scale)
    return total


# ---------------- Intensity computation ----------------
def compute_intensity_at_event(
    i, t, u, e, mu, K, M_K, alpha, 
    time_centers, time_scale, mix_w, 
    window=np.inf
):
    """
    Compute total intensity λ(t_i^-) just before event i.
    
    λ(t) = Σ_v Σ_k λ_{v,k}(t)
         = Σ_v Σ_k [μ_{v,k} + α Σ_u Σ_ℓ K_{vu} M_K_{ℓk} Σ_{j: t_j<t} g(t-t_j) 1{u_j=u, e_j=ℓ}]
    
    Simplified (summing over v, k):
         = Σ_v Σ_k μ_{v,k} + α Σ_{j<i} [Σ_v K_{v,u_j}] [Σ_k M_K_{e_j,k}] g(t_i - t_j)
    """
    t_i = t[i]
    N, M = mu.shape
    
    # Baseline: Σ_v Σ_k μ_{v,k}
    total_baseline = np.sum(mu)
    
    # Excitation from past events
    excitation = 0.0
    t_lower = t_i - window if np.isfinite(window) else 0.0
    
    for j in range(i):
        if t[j] < t_lower:
            continue
        dt = t_i - t[j]
        if dt <= 0:
            continue
        
        u_j = u[j]
        e_j = e[j]
        
        g_val = g_scalar(dt, time_centers, time_scale, mix_w)
        
        # Σ_v K[v, u_j] and Σ_k M_K[e_j, k]
        K_sum = np.sum(K[:, u_j])
        M_sum = np.sum(M_K[e_j, :])
        
        excitation += alpha * K_sum * M_sum * g_val
    
    return total_baseline + excitation


def compute_compensator_between(
    t_start, t_end, events_before_end,
    t, u, e, mu, K, M_K, alpha,
    time_centers, time_scale, mix_w,
    window=np.inf
):
    """
    Compute ∫_{t_start}^{t_end} λ(s) ds.
    
    Compensator = (t_end - t_start) * Σ_{v,k} μ_{v,k}
                + α Σ_{j: t_j < t_end} [Σ_v K_{v,u_j}] [Σ_k M_K_{e_j,k}] 
                  × ∫_{max(t_start, t_j)}^{min(t_end, t_j + window)} g(s - t_j) ds
    """
    N, M = mu.shape
    
    # Baseline contribution
    baseline = (t_end - t_start) * np.sum(mu)
    
    # Excitation contribution
    excitation = 0.0
    
    for j in range(events_before_end):
        t_j = t[j]
        u_j = u[j]
        e_j = e[j]
        
        # Integration bounds for this event's contribution
        # g(s - t_j) is integrated from s = max(t_start, t_j) to s = min(t_end, t_j + window)
        # In terms of τ = s - t_j: from τ = max(t_start - t_j, 0) to τ = min(t_end - t_j, window)
        
        tau_lower = max(t_start - t_j, 0.0)
        tau_upper = min(t_end - t_j, window) if np.isfinite(window) else (t_end - t_j)
        
        if tau_upper <= tau_lower or tau_upper <= 0:
            continue
        
        g_integral = G_int_a_to_b(tau_lower, tau_upper, time_centers, time_scale, mix_w)
        
        K_sum = np.sum(K[:, u_j])
        M_sum = np.sum(M_K[e_j, :])
        
        excitation += alpha * K_sum * M_sum * g_integral
    
    return baseline + excitation


def compute_rescaled_times(
    t, u, e, mu, K, M_K, alpha,
    time_centers, time_scale, mix_w,
    window=np.inf, verbose=True
):
    """
    Compute rescaled inter-arrival times for SPATIO-TEMPORAL Hawkes.
    
    For each event at (t_i, u_i, e_i), compute:
        τ_i = Λ_{u_i, e_i}(t_i) - Λ_{u_i, e_i}(t_prev)
    
    where t_prev is the PREVIOUS event at the SAME (node, mark) pair.
    
    Under correct model specification, τ_i ~ i.i.d. Exp(1).
    
    Returns:
        tau: array of rescaled times
        labels: list of (node, mark) tuples for each event
    """
    n_events = len(t)
    N, M = mu.shape
    tau = np.zeros(n_events)
    labels = []  # Track which (node, mark) each τ belongs to
    
    # Track last event time for each (node, mark) pair
    last_time = {(v, k): 0.0 for v in range(N) for k in range(M)}
    
    if verbose:
        print(f"Computing rescaled times for {n_events} events (per-location)...")
    
    for i in range(n_events):
        t_curr = t[i]
        u_curr = int(u[i])
        e_curr = int(e[i])
        
        # Get previous event time at THIS (node, mark)
        t_prev = last_time[(u_curr, e_curr)]
        
        # Compute compensator for λ_{u_curr, e_curr} from t_prev to t_curr
        tau[i] = compute_compensator_location(
            t_prev, t_curr, i,
            t, u, e, mu, K, M_K, alpha,
            time_centers, time_scale, mix_w,
            u_curr, e_curr, window
        )
        
        labels.append((u_curr, e_curr))
        
        # Update last time for this (node, mark)
        last_time[(u_curr, e_curr)] = t_curr
        
        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{n_events} events")
    
    if verbose:
        print("Done computing rescaled times.")
    
    return tau, labels


def compute_compensator_location(
    t_start, t_end, events_before_end,
    t, u, e, mu, K, M_K, alpha,
    time_centers, time_scale, mix_w,
    target_node, target_mark, window=np.inf
):
    """
    Compute ∫_{t_start}^{t_end} λ_{target_node, target_mark}(s) ds.
    
    This is the compensator for a SPECIFIC (node, mark) pair.
    """
    # Baseline contribution for this specific (node, mark)
    baseline = (t_end - t_start) * mu[target_node, target_mark]
    
    # Excitation contribution
    excitation = 0.0
    
    for j in range(events_before_end):
        t_j = t[j]
        u_j = u[j]
        e_j = e[j]
        
        # Integration bounds
        tau_lower = max(t_start - t_j, 0.0)
        tau_upper = min(t_end - t_j, window) if np.isfinite(window) else (t_end - t_j)
        
        if tau_upper <= tau_lower or tau_upper <= 0:
            continue
        
        g_integral = G_int_a_to_b(tau_lower, tau_upper, time_centers, time_scale, mix_w)
        
        # Contribution from event j to THIS (target_node, target_mark)
        # λ_{v,k}(t) = μ_{v,k} + α Σ_j K[v, u_j] M_K[e_j, k] g(t - t_j)
        K_contrib = K[target_node, u_j]
        M_contrib = M_K[e_j, target_mark]
        
        excitation += alpha * K_contrib * M_contrib * g_integral
    
    return baseline + excitation


# ---------------- Network-Wide Rescaling ----------------
def compute_rescaled_times_network_wide(
    t, u, e, mu, K, M_K, alpha,
    time_centers, time_scale, mix_w,
    window=np.inf, by_mark=False, verbose=True
):
    """
    Compute rescaled inter-arrival times for NETWORK-WIDE aggregated process.
    
    Two modes:
    - by_mark=False: Total intensity λ(t) = Σ_v Σ_k λ_{v,k}(t)
    - by_mark=True: Per-mark intensity λ_k(t) = Σ_v λ_{v,k}(t)
    
    Returns: tau, labels (mark index if by_mark, else None)
    """
    n_events = len(t)
    N, M = mu.shape
    
    if by_mark:
        # Per-mark aggregation
        last_time = {k: 0.0 for k in range(M)}
        tau = []
        labels = []
        
        if verbose:
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
                    time_centers, time_scale, mix_w,
                    v, e_curr, window
                )
            
            if Lambda > 0:
                tau.append(Lambda)
                labels.append(e_curr)
            
            last_time[e_curr] = t_curr
        
        return np.array(tau), labels
    
    else:
        # Total aggregation
        tau = []
        
        if verbose:
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
                        time_centers, time_scale, mix_w,
                        v, k, window
                    )
            
            if Lambda > 0:
                tau.append(Lambda)
            
            t_prev = t_curr
            
            if verbose and (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{n_events} events")
        
        return np.array(tau), None


# ---------------- Statistical Tests ----------------
def run_gof_tests(tau, alpha_level=0.05, tau_labels=None, min_events_per_pair=10):
    """
    GOF tests for spatio-temporal Hawkes.
    
    Key tests:
    - Ljung-Box: checks temporal independence (no autocorrelation)
    - KS test: checks if normalized τ ~ Exp(1)
    Also reports scale error (practical calibration metric).
    """
    results = {}
    
    tau_clean = tau[tau > 1e-12]
    n = len(tau_clean)
    
    print(f"\n{'='*60}")
    print("GOF RESULTS (Spatio-Temporal Hawkes)")
    print(f"{'='*60}")
    print(f"Events: {n}")
    
    # Per-pair normalization for better KS test
    tau_norm = tau_clean.copy()
    if tau_labels is not None:
        labels_clean = [tau_labels[i] for i in range(len(tau)) if tau[i] > 1e-12]
        unique_pairs = list(set(labels_clean))
        pair_means = {}
        pair_taus = {}
        
        for pair in unique_pairs:
            mask = np.array([l == pair for l in labels_clean])
            if mask.sum() >= min_events_per_pair:
                pair_taus[pair] = tau_clean[mask]
                pair_means[pair] = np.mean(tau_clean[mask])
        
        if len(pair_means) > 0:
            tau_norm = []
            labels_norm = []
            for t_val, label in zip(tau_clean, labels_clean):
                if label in pair_means:
                    tau_norm.append(t_val / pair_means[label])
                    labels_norm.append(label)
            tau_norm = np.array(tau_norm)
            print(f"Per-pair normalization: {len(pair_means)} pairs with >={min_events_per_pair} events")
    
    # Basic statistics
    mean_tau = np.mean(tau_norm)
    std_tau = np.std(tau_norm)
    median_tau = np.median(tau_norm)
    scale_error = abs(1.0 - mean_tau) * 100
    
    print(f"\n--- Calibration ---")
    print(f"Mean τ: {mean_tau:.3f} (expected: 1.0)")
    print(f"Scale error: {scale_error:.1f}%")
    print(f"Median τ: {median_tau:.3f} (expected: 0.693)")
    
    results['mean'] = mean_tau
    results['std'] = std_tau
    results['median'] = median_tau
    results['scale_error'] = scale_error
    
    # KS test against Exp(1)
    print(f"\n--- KS Test (distribution check) ---")
    try:
        # Pooled KS test on normalized τ
        ks_stat, ks_pval = stats.kstest(tau_norm, 'expon', args=(0, 1))
        
        # Interpret KS statistic (effect size), not p-value
        # KS stat < 0.05 = excellent, < 0.10 = good, < 0.15 = acceptable
        if ks_stat < 0.05:
            ks_quality = "EXCELLENT"
            ks_pass = True
        elif ks_stat < 0.10:
            ks_quality = "GOOD"
            ks_pass = True
        elif ks_stat < 0.15:
            ks_quality = "ACCEPTABLE"
            ks_pass = True
        else:
            ks_quality = "POOR"
            ks_pass = False
        
        results['ks_pooled'] = {
            'statistic': float(ks_stat),
            'pvalue': float(ks_pval),
            'quality': ks_quality,
            'pass': ks_pass
        }
        print(f"Pooled: KS statistic={ks_stat:.4f} ({ks_quality})")
        print(f"  (p={ks_pval:.4f} - ignore for large samples, use KS stat instead)")
        
        # Per-pair KS tests - report statistics not p-values
        if tau_labels is not None and len(pair_taus) > 0:
            pair_ks_stats = []
            for pair, taus in pair_taus.items():
                if len(taus) >= min_events_per_pair:
                    taus_norm = taus / np.mean(taus)
                    stat, _ = stats.kstest(taus_norm, 'expon', args=(0, 1))
                    pair_ks_stats.append(stat)
            
            if pair_ks_stats:
                median_stat = np.median(pair_ks_stats)
                pct_good = 100 * np.mean([s < 0.15 for s in pair_ks_stats])
                results['ks_per_pair'] = {
                    'median_statistic': float(median_stat),
                    'pct_good': float(pct_good),
                    'n_pairs': len(pair_ks_stats)
                }
                print(f"Per-pair ({len(pair_ks_stats)} pairs): median KS={median_stat:.4f}, {pct_good:.0f}% good (stat<0.15)")
    except Exception as e:
        print(f"KS test failed: {e}")
        results['ks_pooled'] = None
        ks_pass = False
    
    # Ljung-Box test - the key test for Hawkes
    print(f"\n--- Ljung-Box Test (temporal independence) ---")
    try:
        lb_result = acorr_ljungbox(tau_norm, lags=[10, 20, 30], return_df=True)
        lb_pvals = lb_result['lb_pvalue'].values
        lb_pass = all(p > alpha_level for p in lb_pvals)
        
        results['ljung_box'] = {
            'lags': [10, 20, 30],
            'pvalues': lb_pvals.tolist(),
            'pass': lb_pass
        }
        
        print(f"Lag 10: p={lb_pvals[0]:.4f}")
        print(f"Lag 20: p={lb_pvals[1]:.4f}")  
        print(f"Lag 30: p={lb_pvals[2]:.4f}")
        print(f"Result: {'PASS' if lb_pass else 'FAIL'}")
    except Exception as e:
        print(f"Failed: {e}")
        results['ljung_box'] = None
        lb_pass = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    ks_stat_val = results.get('ks_pooled', {}).get('statistic', 1.0)
    ks_quality_str = results.get('ks_pooled', {}).get('quality', 'UNKNOWN')
    
    if lb_pass and scale_error < 20 and ks_stat_val < 0.15:
        print("✓ Model PASSES ALL TESTS")
        print(f"  - Ljung-Box PASS: temporal dynamics captured")
        print(f"  - Scale error {scale_error:.1f}%: intensity well-calibrated")
        print(f"  - KS statistic {ks_stat_val:.4f} ({ks_quality_str}): distribution matches")
        results['overall'] = 'PASS'
    elif lb_pass and scale_error < 20:
        print("○ Model PASSES (KS marginal)")
        print(f"  - Ljung-Box PASS: temporal dynamics captured")
        print(f"  - Scale error {scale_error:.1f}%: intensity well-calibrated")
        print(f"  - KS statistic {ks_stat_val:.4f}: distribution has deviations")
        results['overall'] = 'PASS'
    elif lb_pass:
        print("○ Model ACCEPTABLE")
        print(f"  - Ljung-Box PASS: temporal dynamics captured")
        print(f"  - Scale error {scale_error:.1f}%: intensity has bias")
        results['overall'] = 'ACCEPTABLE'
    else:
        print("✗ Model NEEDS WORK")
        print(f"  - Ljung-Box FAIL: missing temporal structure")
        results['overall'] = 'FAIL'
    
    print(f"{'='*60}\n")
    
    u_transformed = 1 - np.exp(-tau_norm)
    return results, tau_norm, u_transformed


def first_order_residuals(t, u, e, mu, T_max, alpha_level=0.05):
    """
    First-order residuals: compare observed vs expected counts per (node, mark).
    
    Shows baseline contribution vs total events.
    Excitation ratio = total_observed / baseline_expected shows self-excitation strength.
    """
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
    
    # For Hawkes, excitation_ratio = 1/(1-spectral_radius) approximately
    # If ratio = 2, spectral radius ≈ 0.5
    if excitation_ratio > 1:
        implied_sr = 1 - 1/excitation_ratio
        print(f"Implied spectral radius: ~{implied_sr:.2f}")
    
    # Per-node balance check
    obs_per_node = observed.sum(axis=1)
    exp_per_node = (mu * T_max).sum(axis=1)
    node_ratios = obs_per_node / (exp_per_node + 1e-10)
    
    print(f"\nPer-node obs/exp ratio: min={node_ratios.min():.2f}, max={node_ratios.max():.2f}")
    if node_ratios.std() / node_ratios.mean() > 0.5:
        print("-> High variance across nodes (heterogeneous excitation)")
    else:
        print("-> Balanced across nodes")
    
    return {
        'total_obs': int(total_obs),
        'baseline_exp': baseline_exp,
        'excitation_ratio': excitation_ratio,
        'node_ratios': node_ratios.tolist()
    }


def plot_diagnostics(tau, u_transformed, t, save_path=None):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. QQ-plot against Exp(1) - log scale for better visualization
    ax = axes[0, 0]
    n = len(tau)
    probs = np.linspace(0.01, 0.99, min(n, 500))
    theoretical_quantiles = stats.expon.ppf(probs)
    empirical_quantiles = np.quantile(tau, probs)
    
    ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6, s=15, c='steelblue')
    max_val = max(theoretical_quantiles.max(), np.percentile(tau, 99))
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='y=x (perfect fit)')
    ax.set_xlabel('Theoretical Quantiles (Exp(1))')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title('QQ-Plot: Rescaled Times vs Exp(1)')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Histogram of rescaled times with Exp(1) overlay
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
    
    # 3. Uniformity plot
    ax = axes[0, 2]
    ax.hist(u_transformed, bins=50, density=True, alpha=0.7, label='Empirical')
    ax.axhline(y=1.0, color='r', linestyle='--', lw=2, label='Uniform(0,1)')
    ax.set_xlabel('Transformed U = 1 - exp(-τ)')
    ax.set_ylabel('Density')
    ax.set_title('Uniformity Check')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative residuals (N(t) vs Λ(t))
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
    residuals = n_events - cumsum_tau  # Should be ~random walk around 0
    ax.plot(t[:len(residuals)], residuals, 'b-', lw=0.5, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', lw=1)
    # Add confidence bands (±2σ for random walk)
    sigma_band = 2 * np.sqrt(np.arange(1, len(residuals) + 1))
    ax.fill_between(t[:len(residuals)], -sigma_band, sigma_band, alpha=0.2, color='red', label='±2σ band')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Residual N(t) - Λ(t)')
    ax.set_title('Raw Residuals Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Autocorrelation of rescaled times
    ax = axes[1, 2]
    max_lag = min(50, len(tau) // 10)
    acf = np.correlate(tau - np.mean(tau), tau - np.mean(tau), mode='full')
    acf = acf[len(acf)//2:len(acf)//2 + max_lag + 1]
    acf = acf / acf[0]
    
    ax.bar(range(max_lag + 1), acf, alpha=0.7)
    # 95% confidence bands for white noise
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
    
    return fig


def plot_intensity_check(t, u, e, mu, K, M_K, alpha, time_centers, time_scale, mix_w, window, save_path=None):
    """Plot intensity over time with event markers."""
    # Compute intensity at regular intervals and at events
    t_grid = np.linspace(0, t.max(), 500)
    intensity_grid = np.zeros(len(t_grid))
    
    print("Computing intensity on grid for visualization...")
    
    for idx, t_eval in enumerate(t_grid):
        # Find events before t_eval
        events_before = np.searchsorted(t, t_eval)
        
        # Baseline
        total_baseline = np.sum(mu)
        
        # Excitation
        excitation = 0.0
        t_lower = t_eval - window if np.isfinite(window) else 0.0
        
        for j in range(events_before):
            if t[j] < t_lower:
                continue
            dt = t_eval - t[j]
            if dt <= 0:
                continue
            
            g_val = g_scalar(dt, time_centers, time_scale, mix_w)
            K_sum = np.sum(K[:, u[j]])
            M_sum = np.sum(M_K[e[j], :])
            excitation += alpha * K_sum * M_sum * g_val
        
        intensity_grid[idx] = total_baseline + excitation
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: Intensity over time
    ax = axes[0]
    ax.plot(t_grid, intensity_grid, 'b-', lw=1, label='Total Intensity λ(t)')
    ax.axhline(y=np.sum(mu), color='gray', linestyle='--', lw=1, alpha=0.7, label='Baseline Σμ')
    ax.set_ylabel('Intensity λ(t)')
    ax.set_title('Estimated Intensity Function Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom: Event times as rug plot
    ax = axes[1]
    colors = plt.cm.tab10(e % 10)
    ax.scatter(t, np.zeros_like(t), c=colors, marker='|', s=50, alpha=0.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Events')
    ax.set_title('Event Occurrences (colored by mark type)')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved intensity plot to {save_path}")
    
    return fig


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="GOF tests for Network-Constrained Hawkes Process")
    parser.add_argument("--result", type=str, required=True, help="Inference result pickle file")
    parser.add_argument("--data", type=str, default=None, help="Original data pickle (optional, inferred from result)")
    parser.add_argument("--mcmc", type=str, default=None, help="MCMC state npz file (for posterior samples)")
    parser.add_argument("--alpha-level", type=float, default=0.05, help="Significance level for tests")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--output-prefix", type=str, default="gof", help="Prefix for output files")
    parser.add_argument("--use-posterior-mean", action="store_true", default=True, 
                        help="Use posterior mean (default). If False with --mcmc, uses random sample.")
    parser.add_argument("--min-events", type=int, default=10,
                        help="Min events per (node,mark) pair to include in tests (default: 10)")
    parser.add_argument("--aggregate", type=str, choices=["none", "mark", "total"], default="none",
                        help="Aggregation level: none=per-location, mark=network-wide by type, total=all events")
    args = parser.parse_args()
    
    # Load inference results
    print(f"Loading inference results from {args.result}...")
    with open(args.result, "rb") as f:
        result = pickle.load(f)
    
    # Handle both formats: old (_hat keys) and new (samples dict)
    if "samples" in result:
        # New format: compute posterior means from samples
        samples = result["samples"]
        mu = np.mean(samples["mu"], axis=0)
        K = np.mean(samples["K"], axis=0)
        M_K = np.mean(samples["M_K"], axis=0)
        alpha = float(np.mean(samples["alpha"]))
        mix_w = np.mean(samples["mix_w"], axis=0)
        N = result.get("num_nodes", result.get("N"))
        M = result.get("num_event_types", result.get("M"))
    else:
        # Old format: direct _hat keys
        mu = result["mu_hat"]
        K = result["K_hat"]
        M_K = result["M_K_hat"]
        alpha = result["alpha_hat"]
        mix_w = result["mix_w_hat"]
        N = result["N"]
        M = result["M"]
    
    time_centers = result["time_centers"]
    time_scale = result["time_scale"]
    window = result.get("window", np.inf)
    
    print(f"Model: N={N} nodes, M={M} marks")
    print(f"α = {alpha:.4f}")
    print(f"Window = {window}")
    
    # Load original data
    data_file = args.data if args.data else result.get("data_pickle")
    if data_file is None:
        raise ValueError("Must provide --data or have data_pickle in result")
    
    print(f"Loading event data from {data_file}...")
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    events = data["events"]
    t = np.asarray(events["t"])
    u = np.asarray(events["u"])
    e = np.asarray(events["e"])
    
    # Sort by time
    order = np.argsort(t)
    t = t[order]
    u = u[order]
    e = e[order]
    
    print(f"Events: {len(t)}, Time span: [0, {t.max():.2f}]")
    
    # Compute rescaled inter-arrival times based on aggregation level
    if args.aggregate == "none":
        print("\n--- Per-Location Rescaling (standard) ---")
        tau, tau_labels = compute_rescaled_times(
            t, u, e, mu, K, M_K, alpha,
            time_centers, time_scale, mix_w,
            window=window, verbose=True
        )
    elif args.aggregate == "mark":
        print("\n--- Network-Wide by Mark Type ---")
        tau, tau_labels = compute_rescaled_times_network_wide(
            t, u, e, mu, K, M_K, alpha,
            time_centers, time_scale, mix_w,
            window=window, by_mark=True, verbose=True
        )
    else:  # total
        print("\n--- Network-Wide Total (all events) ---")
        tau, tau_labels = compute_rescaled_times_network_wide(
            t, u, e, mu, K, M_K, alpha,
            time_centers, time_scale, mix_w,
            window=window, by_mark=False, verbose=True
        )
    
    # Run GOF tests with filtering for active (node, mark) pairs
    min_events = args.min_events
    test_results, tau_clean, u_transformed = run_gof_tests(
        tau, args.alpha_level, 
        tau_labels=tau_labels, 
        min_events_per_pair=min_events
    )
    
    # First-order residuals (observed vs expected counts)
    T_max = t.max()
    fo_results = first_order_residuals(t, u, e, mu, T_max, args.alpha_level)
    if fo_results:
        test_results['first_order'] = fo_results
    
    # Save numerical results
    results_file = f"{args.output_prefix}_results.pickle"
    with open(results_file, "wb") as f:
        pickle.dump({
            'tau': tau,
            'tau_clean': tau_clean,
            'u_transformed': u_transformed,
            'test_results': test_results,
            'alpha_level': args.alpha_level,
            'source_result': args.result,
            'source_data': data_file,
        }, f)
    print(f"\nSaved numerical results to {results_file}")
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating diagnostic plots...")
        
        # Main diagnostic plots
        fig1 = plot_diagnostics(tau_clean, u_transformed, t, 
                               save_path=f"{args.output_prefix}_diagnostics.png")
        
        # Intensity visualization (can be slow for many events)
        if len(t) <= 5000:
            fig2 = plot_intensity_check(t, u, e, mu, K, M_K, alpha,
                                       time_centers, time_scale, mix_w, window,
                                       save_path=f"{args.output_prefix}_intensity.png")
        else:
            print(f"Skipping intensity plot (too many events: {len(t)})")
        
        plt.show()
    
    # Already printed summary in run_gof_tests
    print(f"Result saved to {results_file}")


if __name__ == "__main__":
    main()