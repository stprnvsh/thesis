#!/usr/bin/env python3
"""
Evaluate Hawkes model predictions vs actual data.

Splits data at a cutoff time (default 4pm), uses events before as "history",
predicts intensity after, and compares with actual event counts.

Metrics: R², MAE, RMSE, Pearson correlation, MAPE
"""

import argparse
import pickle
import numpy as np
from scipy import stats


def load_model(result_file):
    """Load trained model parameters (posterior means from MCMC samples)."""
    with open(result_file, 'rb') as f:
        result = pickle.load(f)
    
    # Check if samples are stored (new format) or direct values (old format)
    if 'samples' in result:
        samples = result['samples']
        mu = np.mean(samples['mu'], axis=0)
        alpha = np.mean(samples['alpha'])
        mix_w = np.mean(samples['mix_w'], axis=0)
        K = np.mean(samples['K'], axis=0)
        M_K = np.mean(samples.get('M_K', np.ones((1, 2, 2)) * 0.5), axis=0)
    else:
        mu = result.get('mu_hat', result.get('mu'))
        alpha = result.get('alpha_hat', result.get('alpha'))
        mix_w = result.get('mix_w_hat', result.get('mix_w'))
        K = result.get('K_hat', result.get('K'))
        M_K = result.get('M_K_hat', result.get('M_K', np.ones((2, 2)) * 0.5))
    
    time_centers = result.get('time_centers', result.get('time_centers_np'))
    time_scale = result.get('time_scale', 0.1)
    window = result.get('window', 1.0)
    
    N = result.get('num_nodes', mu.shape[0] if mu is not None else 12)
    M = result.get('num_event_types', mu.shape[1] if mu is not None and len(mu.shape) > 1 else 2)
    
    return {
        'mu': mu, 'alpha': alpha, 'mix_w': mix_w,
        'time_centers': time_centers, 'time_scale': time_scale,
        'K': K, 'M_K': M_K, 'window': window, 'N': N, 'M': M
    }


def load_events(data_file):
    """Load event data."""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    events = data['events']
    t = np.array(events['t'])
    u = np.array(events['u'])
    e = np.array(events['e'])
    
    # Sort by time
    order = np.argsort(t)
    return t[order], u[order], e[order]


def temporal_kernel(tau, centers, scale, weights):
    """Compute temporal kernel g(τ) as mixture of Gaussians."""
    if tau < 0:
        return 0.0
    
    g = 0.0
    for c, w in zip(centers, weights):
        g += w * np.exp(-0.5 * ((tau - c) / scale) ** 2)
    
    # Normalize (approximate)
    return g


def compute_intensity(t_eval, history_t, history_u, history_e, model, target_node, target_mark):
    """
    Compute λ_{target_node, target_mark}(t_eval) given event history.
    """
    mu = model['mu']
    alpha = model['alpha']
    K = model['K']
    M_K = model['M_K']
    window = model['window']
    centers = model['time_centers']
    scale = model['time_scale']
    weights = model['mix_w']
    
    # Baseline
    if len(mu.shape) == 1:
        lam = mu[target_node]
    else:
        lam = mu[target_node, target_mark]
    
    # Excitation from history
    for j in range(len(history_t)):
        t_j = history_t[j]
        tau = t_eval - t_j
        
        if tau <= 0 or tau > window:
            continue
        
        u_j = int(history_u[j])
        e_j = int(history_e[j])
        
        g_tau = temporal_kernel(tau, centers, scale, weights)
        K_contrib = K[target_node, u_j] if K is not None else 1.0 / model['N']
        M_contrib = M_K[e_j, target_mark] if M_K is not None else 0.5
        
        lam += alpha * K_contrib * M_contrib * g_tau
    
    return max(lam, 0.0)


def predict_counts(t_bins, all_t, all_u, all_e, model, N, M):
    """
    Predict expected event counts in each time bin for each (node, mark).
    
    Uses ALL events before each bin's midpoint as history (including test events).
    This is a fair comparison: at prediction time t, we know all events before t.
    
    Returns: (n_bins, N, M) array of predicted counts
    """
    n_bins = len(t_bins) - 1
    predicted = np.zeros((n_bins, N, M))
    
    for b in range(n_bins):
        t_start = t_bins[b]
        t_end = t_bins[b + 1]
        t_mid = (t_start + t_end) / 2
        dt = t_end - t_start
        
        # Use all events before t_mid as history
        mask = all_t < t_mid
        hist_t = all_t[mask]
        hist_u = all_u[mask]
        hist_e = all_e[mask]
        
        for v in range(N):
            for k in range(M):
                lam = compute_intensity(t_mid, hist_t, hist_u, hist_e, model, v, k)
                predicted[b, v, k] = lam * dt
    
    return predicted


def count_events(t, u, e, t_bins, N, M):
    """Count actual events in each time bin for each (node, mark)."""
    n_bins = len(t_bins) - 1
    counts = np.zeros((n_bins, N, M))
    
    for i in range(len(t)):
        bin_idx = np.searchsorted(t_bins, t[i]) - 1
        if 0 <= bin_idx < n_bins:
            counts[bin_idx, int(u[i]), int(e[i])] += 1
    
    return counts


def exponential_moving_average(series, alpha=0.4):
    """
    EMA smoothing (no future leakage).
    alpha: 0.1=smooth, 0.5=responsive
    """
    if len(series) == 0:
        return series
    ema = np.zeros_like(series, dtype=float)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
    return ema


def compute_metrics(actual, predicted):
    """Compute evaluation metrics."""
    actual_flat = actual.flatten()
    pred_flat = predicted.flatten()
    
    # Filter out zeros for some metrics
    nonzero_mask = actual_flat > 0
    
    # Basic metrics
    mae = np.mean(np.abs(actual_flat - pred_flat))
    rmse = np.sqrt(np.mean((actual_flat - pred_flat) ** 2))
    
    # R² (coefficient of determination)
    ss_res = np.sum((actual_flat - pred_flat) ** 2)
    ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Pearson correlation
    if np.std(actual_flat) > 0 and np.std(pred_flat) > 0:
        pearson_r, pearson_p = stats.pearsonr(actual_flat, pred_flat)
    else:
        pearson_r, pearson_p = 0, 1
    
    # Spearman correlation (rank-based)
    if len(actual_flat) > 2:
        spearman_r, spearman_p = stats.spearmanr(actual_flat, pred_flat)
    else:
        spearman_r, spearman_p = 0, 1
    
    # MAPE (only where actual > 0)
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(actual_flat[nonzero_mask] - pred_flat[nonzero_mask]) / actual_flat[nonzero_mask]) * 100
    else:
        mape = float('inf')
    
    # Total counts
    total_actual = actual_flat.sum()
    total_pred = pred_flat.sum()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mape': mape,
        'total_actual': total_actual,
        'total_pred': total_pred
    }


def compute_metrics_with_smoothing(actual, predicted, N, M, ema_alpha=0.4):
    """
    Compute metrics with EMA smoothing and per-node rescaling.
    
    For each node: rescale predictions to match observed mean, then smooth with EMA.
    """
    n_bins = actual.shape[0]
    
    all_obs_smooth = []
    all_pred_rescaled = []
    
    for node in range(N):
        # Sum across marks for this node
        obs_node = actual[:, node, :].sum(axis=1) if len(actual.shape) == 3 else actual[:, node]
        pred_node = predicted[:, node, :].sum(axis=1) if len(predicted.shape) == 3 else predicted[:, node]
        
        # Skip empty nodes
        if obs_node.sum() == 0 and pred_node.mean() < 1e-9:
            continue
        
        # Per-node rescaling (match means)
        scale = obs_node.mean() / pred_node.mean() if pred_node.mean() > 0 else 1.0
        pred_rescaled = pred_node * scale
        
        # EMA smoothing
        obs_smooth = exponential_moving_average(obs_node, ema_alpha)
        
        all_obs_smooth.extend(obs_smooth)
        all_pred_rescaled.extend(pred_rescaled)
    
    if not all_obs_smooth:
        return None
    
    all_obs_smooth = np.array(all_obs_smooth)
    all_pred_rescaled = np.array(all_pred_rescaled)
    
    # Compute metrics on smoothed/rescaled data
    return compute_metrics(all_obs_smooth, all_pred_rescaled)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hawkes predictions')
    parser.add_argument('--result', required=True, help='Model result pickle file')
    parser.add_argument('--data', required=True, help='Event data pickle file')
    parser.add_argument('--cutoff', type=float, default=16.0, help='Cutoff time in hours (default: 16 = 4pm)')
    parser.add_argument('--bin-size', type=float, default=0.5, help='Time bin size in hours (default: 0.5 = 30min)')
    parser.add_argument('--aggregate', choices=['none', 'node', 'mark', 'time'], default='none',
                        help='Aggregation level for metrics')
    parser.add_argument('--ema-alpha', type=float, default=0.4,
                        help='EMA smoothing factor (0.1=smooth, 0.5=responsive, default: 0.4)')
    parser.add_argument('--per-node-rescale', action='store_true',
                        help='Rescale predictions per node to match observed mean')
    parser.add_argument('--output', type=str, default='prediction_eval.pickle', help='Output file')
    args = parser.parse_args()
    
    print(f"Loading model from {args.result}...")
    model = load_model(args.result)
    print(f"  N={model['N']} nodes, M={model['M']} marks")
    print(f"  α={model['alpha']:.4f}, window={model['window']}h")
    
    print(f"\nLoading events from {args.data}...")
    t, u, e = load_events(args.data)
    print(f"  Total events: {len(t)}")
    print(f"  Time span: {t.min():.2f}h - {t.max():.2f}h")
    
    # Split at cutoff
    cutoff = args.cutoff
    train_mask = t < cutoff
    test_mask = t >= cutoff
    
    print(f"\n--- Data Split at {cutoff}h (4pm) ---")
    print(f"Training events (history): {train_mask.sum()}")
    print(f"Test events: {test_mask.sum()}")
    
    # Create time bins for test period
    t_max = t.max()
    t_bins = np.arange(cutoff, t_max + args.bin_size, args.bin_size)
    n_bins = len(t_bins) - 1
    print(f"Test period: {cutoff}h - {t_max:.2f}h ({n_bins} bins of {args.bin_size}h)")
    
    # Count actual events in test period
    print("\nCounting actual events...")
    actual = count_events(t, u, e, t_bins, model['N'], model['M'])
    
    # Predict counts (uses all events before each prediction time)
    print("Computing predictions (this may take a moment)...")
    predicted = predict_counts(t_bins, t, u, e, model, model['N'], model['M'])
    
    # Aggregation options
    if args.aggregate == 'node':
        # Sum across marks: (n_bins, N)
        actual_agg = actual.sum(axis=2)
        predicted_agg = predicted.sum(axis=2)
        print(f"Aggregating across marks: {actual.shape} -> {actual_agg.shape}")
    elif args.aggregate == 'mark':
        # Sum across nodes: (n_bins, M)
        actual_agg = actual.sum(axis=1)
        predicted_agg = predicted.sum(axis=1)
        print(f"Aggregating across nodes: {actual.shape} -> {actual_agg.shape}")
    elif args.aggregate == 'time':
        # Sum across time: (N, M)
        actual_agg = actual.sum(axis=0)
        predicted_agg = predicted.sum(axis=0)
        print(f"Aggregating across time: {actual.shape} -> {actual_agg.shape}")
    else:
        actual_agg = actual
        predicted_agg = predicted
    
    # Compute metrics
    metrics = compute_metrics(actual_agg, predicted_agg)
    
    # Also compute with EMA + per-node rescaling
    if args.per_node_rescale:
        metrics_smooth = compute_metrics_with_smoothing(
            actual, predicted, model['N'], model['M'], args.ema_alpha
        )
    else:
        metrics_smooth = None
    
    print(f"\n{'='*60}")
    print("PREDICTION EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Test period: {cutoff}h - {t_max:.2f}h")
    print(f"Bins: {n_bins} × {args.bin_size}h = {n_bins * args.bin_size:.1f}h")
    print(f"\n--- Counts ---")
    print(f"Actual events: {int(metrics['total_actual'])}")
    print(f"Predicted events: {metrics['total_pred']:.1f}")
    print(f"Ratio: {metrics['total_pred']/metrics['total_actual']:.2f}")
    
    print(f"\n--- Error Metrics ---")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.1f}%")
    
    print(f"\n--- Correlation Metrics ---")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
    print(f"Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")
    
    # Smoothed/rescaled metrics
    if metrics_smooth is not None:
        print(f"\n--- With EMA Smoothing + Per-Node Rescaling (α={args.ema_alpha}) ---")
        print(f"R²: {metrics_smooth['r2']:.4f}")
        print(f"Pearson r: {metrics_smooth['pearson_r']:.4f}")
        print(f"MAE: {metrics_smooth['mae']:.4f}")
    
    # Network-wide by type (most useful metric)
    print(f"\n--- Network-Wide by Type (sum across nodes) ---")
    actual_by_type = actual.sum(axis=1)  # (n_bins, M)
    pred_by_type = predicted.sum(axis=1)
    
    # Raw
    a_flat = actual_by_type.flatten()
    p_flat = pred_by_type.flatten()
    r2_raw = 1 - np.sum((a_flat - p_flat)**2) / np.sum((a_flat - a_flat.mean())**2)
    pearson_raw, _ = stats.pearsonr(a_flat, p_flat)
    print(f"Raw:      R²={r2_raw:.4f}, Pearson={pearson_raw:.4f}")
    
    # With EMA
    a_smooth = np.zeros_like(actual_by_type, dtype=float)
    p_smooth = np.zeros_like(pred_by_type, dtype=float)
    for m in range(model['M']):
        a_smooth[:, m] = exponential_moving_average(actual_by_type[:, m], args.ema_alpha)
        p_smooth[:, m] = exponential_moving_average(pred_by_type[:, m], args.ema_alpha)
    a_flat = a_smooth.flatten()
    p_flat = p_smooth.flatten()
    r2_ema = 1 - np.sum((a_flat - p_flat)**2) / np.sum((a_flat - a_flat.mean())**2)
    pearson_ema, _ = stats.pearsonr(a_flat, p_flat)
    print(f"+EMA:     R²={r2_ema:.4f}, Pearson={pearson_ema:.4f}")
    
    # Network-wide total
    actual_total = actual.sum(axis=(1, 2))  # (n_bins,)
    pred_total = predicted.sum(axis=(1, 2))
    r2_total = 1 - np.sum((actual_total - pred_total)**2) / np.sum((actual_total - actual_total.mean())**2)
    a_total_ema = exponential_moving_average(actual_total, args.ema_alpha)
    p_total_ema = exponential_moving_average(pred_total, args.ema_alpha)
    r2_total_ema = 1 - np.sum((a_total_ema - p_total_ema)**2) / np.sum((a_total_ema - a_total_ema.mean())**2)
    print(f"\n--- Network-Wide Total (all events) ---")
    print(f"Raw:      R²={r2_total:.4f}")
    print(f"+EMA:     R²={r2_total_ema:.4f}")
    
    # Per-node metrics (spatial prediction quality)
    print(f"\n--- Per-Node (spatial prediction) ---")
    actual_per_node = actual.sum(axis=2)  # (n_bins, N) - sum over marks
    pred_per_node = predicted.sum(axis=2)
    
    # Overall per-node R²
    a_flat = actual_per_node.flatten()
    p_flat = pred_per_node.flatten()
    r2_node_raw = 1 - np.sum((a_flat - p_flat)**2) / np.sum((a_flat - a_flat.mean())**2)
    pearson_node_raw, _ = stats.pearsonr(a_flat, p_flat)
    
    # With EMA per node
    a_node_ema = np.zeros_like(actual_per_node, dtype=float)
    p_node_ema = np.zeros_like(pred_per_node, dtype=float)
    for n in range(model['N']):
        a_node_ema[:, n] = exponential_moving_average(actual_per_node[:, n], args.ema_alpha)
        p_node_ema[:, n] = exponential_moving_average(pred_per_node[:, n], args.ema_alpha)
    a_flat = a_node_ema.flatten()
    p_flat = p_node_ema.flatten()
    r2_node_ema = 1 - np.sum((a_flat - p_flat)**2) / np.sum((a_flat - a_flat.mean())**2)
    pearson_node_ema, _ = stats.pearsonr(a_flat, p_flat)
    print(f"Raw:      R²={r2_node_raw:.4f}, Pearson={pearson_node_raw:.4f}")
    print(f"+EMA:     R²={r2_node_ema:.4f}, Pearson={pearson_node_ema:.4f}")
    
    # Per-node breakdown for top active nodes
    node_totals = actual.sum(axis=(0, 2))
    sorted_nodes = np.argsort(node_totals)[::-1]
    
    print(f"\n--- Top Nodes by Activity ---")
    print(f"{'Node':<6} {'Events':<8} {'Pred':<8} {'R²':<8} {'Pearson':<8}")
    print("-" * 40)
    
    for rank, node in enumerate(sorted_nodes[:10]):
        a_node = actual_per_node[:, node]
        p_node = pred_per_node[:, node]
        n_events = int(node_totals[node])
        pred_events = p_node.sum()
        
        if np.var(a_node) > 0:
            r2_n = 1 - np.sum((a_node - p_node)**2) / np.sum((a_node - a_node.mean())**2)
            pearson_n, _ = stats.pearsonr(a_node, p_node) if n_events > 2 else (np.nan, 1)
        else:
            r2_n, pearson_n = np.nan, np.nan
        print(f"{node:<6} {n_events:<8} {pred_events:<8.1f} {r2_n:<8.4f} {pearson_n:<8.4f}")
    
    top5 = sorted_nodes[:5]
    top10 = sorted_nodes[:10]
    a_top5 = actual_per_node[:, top5].flatten()
    p_top5 = pred_per_node[:, top5].flatten()
    r2_top5 = 1 - np.sum((a_top5 - p_top5)**2) / np.sum((a_top5 - a_top5.mean())**2)
    a_top10 = actual_per_node[:, top10].flatten()
    p_top10 = pred_per_node[:, top10].flatten()
    r2_top10 = 1 - np.sum((a_top10 - p_top10)**2) / np.sum((a_top10 - a_top10.mean())**2)
    print("-" * 40)
    print(f"Top 5 nodes combined R²:  {r2_top5:.4f}")
    print(f"Top 10 nodes combined R²: {r2_top10:.4f}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    if metrics['r2'] > 0.7:
        print("✓ R² > 0.7: Excellent predictive power")
    elif metrics['r2'] > 0.5:
        print("○ R² 0.5-0.7: Good predictive power")
    elif metrics['r2'] > 0.3:
        print("○ R² 0.3-0.5: Moderate predictive power")
    else:
        print("✗ R² < 0.3: Weak predictive power")
    
    if metrics['pearson_r'] > 0.8:
        print("✓ Pearson > 0.8: Strong correlation")
    elif metrics['pearson_r'] > 0.6:
        print("○ Pearson 0.6-0.8: Moderate correlation")
    else:
        print("✗ Pearson < 0.6: Weak correlation")
    
    print(f"{'='*60}\n")
    
    # Save results
    with open(args.output, 'wb') as f:
        pickle.dump({
            'actual': actual,
            'predicted': predicted,
            'metrics': metrics,
            't_bins': t_bins,
            'cutoff': cutoff,
            'bin_size': args.bin_size
        }, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

