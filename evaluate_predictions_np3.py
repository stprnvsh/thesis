#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Evaluation for Joint Spatio-Temporal Hawkes Model (nonpm_window_3.py).
"""

import argparse
import pickle
import numpy as np
from scipy import stats
from scipy.special import erf
from scipy.sparse.csgraph import shortest_path


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
    rt2 = np.sqrt(2.0)
    pref = scale * np.sqrt(np.pi / 2.0)
    return pref * (erf((x - c) / (rt2 * scale)) - erf((-c) / (rt2 * scale)))


def gauss_bump_int_0_to_inf(c, scale):
    rt2 = np.sqrt(2.0)
    return scale * np.sqrt(np.pi / 2.0) * (1.0 - erf((-c) / (rt2 * scale)))


def build_joint_kernel_components(kernel_param, time_centers, time_scale, 
                                   dist_centers, dist_scale, D):
    """Build S_t and denom for the joint kernel."""
    B_t = len(time_centers)
    N = D.shape[0]
    
    w_pos = np.log1p(np.exp(kernel_param)) + 1e-8  # softplus
    Psi_r = np.stack([np.exp(-0.5 * ((D - c) / dist_scale) ** 2) for c in dist_centers], axis=-1)
    S_t = np.tensordot(Psi_r, w_pos, axes=[[2], [1]])
    I_inf = np.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers])
    denom = np.maximum(np.tensordot(S_t, I_inf, axes=[[2], [0]]), 1e-12)
    
    return S_t, denom


def psi_int(dt_cap, v, u_j, S_t, denom, time_centers, time_scale):
    """∫_0^{dt_cap} ψ̃(τ, r_{v,u_j}) dτ"""
    dt_cap = max(dt_cap, 0.0)
    I_cap = np.array([gauss_bump_int_0_to(dt_cap, c, time_scale) for c in time_centers])
    num = np.dot(S_t[v, u_j], I_cap)
    return num / denom[v, u_j]


def predict_counts(t_start, t_end, t_hist, u_hist, e_hist, 
                   mu, K, M_K, alpha, S_t, denom, 
                   time_centers, time_scale, window, n_bins, N, M):
    """Predict event counts in bins."""
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    predicted = np.zeros((n_bins, N, M))
    
    for b in range(n_bins):
        bin_start, bin_end = bin_edges[b], bin_edges[b+1]
        dt_bin = bin_end - bin_start
        
        # History up to bin_start (events that can contribute)
        hist_mask = t_hist < bin_end
        t_h = t_hist[hist_mask]
        u_h = u_hist[hist_mask]
        e_h = e_hist[hist_mask]
        
        for v in range(N):
            for k in range(M):
                # Baseline
                rate = mu[v, k] * dt_bin
                
                # Excitation from history: ∫_{bin_start}^{bin_end} excitation ds
                for j in range(len(t_h)):
                    t_j = t_h[j]
                    if t_j >= bin_end:
                        continue
                    
                    u_j, e_j = u_h[j], e_h[j]
                    
                    # Integration limits relative to event j
                    a = max(bin_start - t_j, 0.0)
                    b_lim = min(bin_end - t_j, window)
                    
                    if b_lim <= a:
                        continue
                    
                    # ∫_a^b ψ̃(τ, r) dτ = psi_int(b) - psi_int(a)
                    int_b = psi_int(b_lim, v, u_j, S_t, denom, time_centers, time_scale)
                    int_a = psi_int(a, v, u_j, S_t, denom, time_centers, time_scale) if a > 0 else 0
                    
                    rate += alpha * K[v, u_j] * M_K[e_j, k] * (int_b - int_a)
                
                predicted[b, v, k] = rate
    
    return predicted, bin_edges


def exponential_moving_average(series, alpha=0.4):
    """Apply EMA smoothing."""
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True, help="Inference result pickle")
    p.add_argument("--data", required=True, help="Data pickle")
    p.add_argument("--split-hour", type=float, default=16.0, help="Train/test split hour")
    p.add_argument("--bin-size", type=float, default=0.25, help="Bin size in hours")
    p.add_argument("--ema-alpha", type=float, default=0.4, help="EMA smoothing factor")
    args = p.parse_args()
    
    # Load
    with open(args.result, "rb") as f:
        result = pickle.load(f)
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    
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
    N, M = result["N"], result["M"]
    
    events = data["events"]
    t = np.array(events["t"])
    u = np.array(events["u"])
    e = np.array(events["e"])
    
    order = np.argsort(t)
    t, u, e = t[order], u[order], e[order]
    
    adjacency = np.array(data["adjacency_matrix"])
    D = compute_network_distance(adjacency)
    
    S_t, denom = build_joint_kernel_components(
        kernel_param, time_centers, time_scale, dist_centers, dist_scale, D
    )
    
    print(f"Loaded {len(t)} events, N={N}, M={M}")
    print(f"Alpha: {alpha:.4f}, Window: {window}h")
    
    # Split
    train_mask = t < args.split_hour
    test_mask = t >= args.split_hour
    
    t_train, u_train, e_train = t[train_mask], u[train_mask], e[train_mask]
    t_test, u_test, e_test = t[test_mask], u[test_mask], e[test_mask]
    
    T_end = t.max()
    n_bins = int((T_end - args.split_hour) / args.bin_size)
    
    print(f"\nTrain: {len(t_train)} events (before {args.split_hour}h)")
    print(f"Test: {len(t_test)} events ({n_bins} bins of {args.bin_size}h)")
    
    # Count actual
    bin_edges = np.linspace(args.split_hour, T_end, n_bins + 1)
    actual = np.zeros((n_bins, N, M))
    for i in range(len(t_test)):
        b = int((t_test[i] - args.split_hour) / args.bin_size)
        if 0 <= b < n_bins:
            actual[b, u_test[i], e_test[i]] += 1
    
    # Predict with rolling history
    print("Computing predictions...")
    t_hist = t_train.copy()
    u_hist = u_train.copy()
    e_hist = e_train.copy()
    
    predicted, _ = predict_counts(
        args.split_hour, T_end, np.concatenate([t_train, t_test]),
        np.concatenate([u_train, u_test]), np.concatenate([e_train, e_test]),
        mu, K, M_K, alpha, S_t, denom, time_centers, time_scale, window,
        n_bins, N, M
    )
    
    # Network-wide by type
    actual_by_type = actual.sum(axis=1)  # (n_bins, M)
    pred_by_type = predicted.sum(axis=1)
    
    r2_raw = []
    r2_ema = []
    for k in range(M):
        a, p = actual_by_type[:, k], pred_by_type[:, k]
        var = np.var(a)
        if var > 0:
            r2_raw.append(1 - np.sum((a - p)**2) / np.sum((a - a.mean())**2))
            a_ema = exponential_moving_average(a, args.ema_alpha)
            p_ema = exponential_moving_average(p, args.ema_alpha)
            r2_ema.append(1 - np.sum((a_ema - p_ema)**2) / np.sum((a_ema - a_ema.mean())**2))
    
    # Network-wide total
    actual_total = actual.sum(axis=(1, 2))
    pred_total = predicted.sum(axis=(1, 2))
    r2_total = 1 - np.sum((actual_total - pred_total)**2) / np.sum((actual_total - actual_total.mean())**2)
    
    a_total_ema = exponential_moving_average(actual_total, args.ema_alpha)
    p_total_ema = exponential_moving_average(pred_total, args.ema_alpha)
    r2_total_ema = 1 - np.sum((a_total_ema - p_total_ema)**2) / np.sum((a_total_ema - a_total_ema.mean())**2)
    
    pearson_raw, _ = stats.pearsonr(actual_total, pred_total)
    pearson_ema, _ = stats.pearsonr(a_total_ema, p_total_ema)
    
    # Per-node
    actual_per_node = actual.sum(axis=2)
    pred_per_node = predicted.sum(axis=2)
    a_flat = actual_per_node.flatten()
    p_flat = pred_per_node.flatten()
    r2_node = 1 - np.sum((a_flat - p_flat)**2) / np.sum((a_flat - a_flat.mean())**2)
    
    print(f"\n{'='*60}")
    print("PREDICTION EVALUATION (Joint Spatio-Temporal Kernel)")
    print(f"{'='*60}")
    print(f"Test period: {args.split_hour}h - {T_end:.1f}h")
    print(f"Bins: {n_bins} × {args.bin_size}h")
    print(f"\nActual events: {int(actual.sum())}")
    print(f"Predicted events: {predicted.sum():.1f}")
    
    print(f"\n--- Network-Wide by Type ---")
    for k in range(M):
        print(f"  Mark {k}: Raw R²={r2_raw[k]:.4f}, +EMA R²={r2_ema[k]:.4f}")
    
    print(f"\n--- Network-Wide Total ---")
    print(f"Raw:   R²={r2_total:.4f}, Pearson={pearson_raw:.4f}")
    print(f"+EMA:  R²={r2_total_ema:.4f}, Pearson={pearson_ema:.4f}")
    
    print(f"\n--- Per-Node ---")
    print(f"R²={r2_node:.4f}")
    
    # Per-node breakdown for top active nodes
    node_totals = actual.sum(axis=(0, 2))  # total events per node
    sorted_nodes = np.argsort(node_totals)[::-1]  # most active first
    
    print(f"\n--- Top Nodes by Activity ---")
    print(f"{'Node':<6} {'Events':<8} {'Pred':<8} {'R²':<8} {'Pearson':<8}")
    print("-" * 40)
    
    node_r2s = []
    for rank, node in enumerate(sorted_nodes[:10]):
        a_node = actual_per_node[:, node]
        p_node = pred_per_node[:, node]
        n_events = int(node_totals[node])
        pred_events = p_node.sum()
        
        if np.var(a_node) > 0:
            r2_n = 1 - np.sum((a_node - p_node)**2) / np.sum((a_node - a_node.mean())**2)
            pearson_n, _ = stats.pearsonr(a_node, p_node) if n_events > 2 else (np.nan, 1)
        else:
            r2_n = np.nan
            pearson_n = np.nan
        
        node_r2s.append((node, n_events, r2_n))
        print(f"{node:<6} {n_events:<8} {pred_events:<8.1f} {r2_n:<8.4f} {pearson_n:<8.4f}")
    
    # Summary for top 5 and top 10
    top5_nodes = sorted_nodes[:5]
    top10_nodes = sorted_nodes[:10]
    
    a_top5 = actual_per_node[:, top5_nodes].flatten()
    p_top5 = pred_per_node[:, top5_nodes].flatten()
    r2_top5 = 1 - np.sum((a_top5 - p_top5)**2) / np.sum((a_top5 - a_top5.mean())**2)
    
    a_top10 = actual_per_node[:, top10_nodes].flatten()
    p_top10 = pred_per_node[:, top10_nodes].flatten()
    r2_top10 = 1 - np.sum((a_top10 - p_top10)**2) / np.sum((a_top10 - a_top10.mean())**2)
    
    print("-" * 40)
    print(f"Top 5 nodes combined R²:  {r2_top5:.4f}")
    print(f"Top 10 nodes combined R²: {r2_top10:.4f}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

