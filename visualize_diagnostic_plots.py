#!/usr/bin/env python3
"""
Diagnostic plots for understanding windowed Hawkes model issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from math import erf, sqrt, pi

def gauss_int_0_to(x, c, s):
    """Integral of Gaussian from 0 to x"""
    rt2 = sqrt(2.0)
    pref = s * sqrt(pi / 2.0)
    return pref * (erf((x - c) / (rt2 * s)) - erf((-c) / (rt2 * s)))

def load_results_and_state(result_file):
    """Load both result and MCMC state if available"""
    with open(result_file, 'rb') as f:
        results = pickle.load(f)
    
    # Try to load MCMC state
    state_file = results.get('mcmc_state_file', f"mcmc_state_np_{result_file.split('.')[0]}.npz")
    try:
        state = np.load(state_file, allow_pickle=True)
        return results, state
    except:
        return results, None

def plot_temporal_kernel_analysis(results, state):
    """Detailed analysis of temporal kernel and window effects"""
    
    # Extract temporal parameters
    time_centers = np.asarray(results['time_centers'])
    time_scale = float(results['time_scale'])
    window = float(results.get('window', np.inf))
    T = float(results['T'])
    
    # Get mixture weights from MCMC state
    if state is not None and 'a_uncon' in state:
        a_uncon = state['a_uncon']
        a_mean = np.mean(a_uncon, axis=0)
        w_pos = np.log1p(np.exp(-np.abs(a_mean))) + np.maximum(a_mean, 0.0) + 1e-8
    else:
        # Fallback to uniform weights
        w_pos = np.ones_like(time_centers)
    
    # Normalize to get mixture weights
    ints = np.array([gauss_int_0_to(np.inf, c, time_scale) for c in time_centers])
    Z = np.dot(w_pos, ints) + 1e-12
    mix_w = w_pos / Z
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Temporal kernel g̃(τ)
    tau = np.linspace(0, min(T, 10*window), 1000)
    g_vals = np.zeros_like(tau)
    for i, t in enumerate(tau):
        if t > 0:
            phi = np.exp(-0.5 * ((t - time_centers) / time_scale) ** 2)
            g_vals[i] = phi @ mix_w
    
    ax = axes[0, 0]
    ax.plot(tau, g_vals, 'b-', linewidth=2)
    if np.isfinite(window):
        ax.axvline(window, color='r', linestyle='--', linewidth=2, label=f'Window W={window:.1f}')
        # Shade the captured region
        mask = tau <= window
        ax.fill_between(tau[mask], 0, g_vals[mask], alpha=0.3, color='green', 
                       label=f'Captured mass: {gauss_int_0_to(window, time_centers, time_scale) @ mix_w:.1%}')
    ax.set_xlabel('Time lag τ (hours)')
    ax.set_ylabel('g̃(τ)')
    ax.set_title('Temporal Kernel (Unit Integral)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution of g̃
    ax = axes[0, 1]
    G_vals = np.zeros_like(tau)
    for i, t in enumerate(tau):
        integrals = np.array([gauss_int_0_to(t, c, time_scale) for c in time_centers])
        G_vals[i] = integrals @ mix_w
    
    ax.plot(tau, G_vals, 'b-', linewidth=2)
    ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    if np.isfinite(window):
        ax.axvline(window, color='r', linestyle='--', linewidth=2)
        window_mass = gauss_int_0_to(window, time_centers, time_scale) @ mix_w
        ax.plot(window, window_mass, 'ro', markersize=10)
        ax.text(window*1.1, window_mass, f'{window_mass:.1%}', fontsize=12)
    ax.set_xlabel('Time lag τ (hours)')
    ax.set_ylabel('∫₀^τ g̃(s) ds')
    ax.set_title('Cumulative Distribution')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # 3. Effective kernel α·g̃(τ) 
    ax = axes[1, 0]
    alpha = float(results['alpha_hat'])
    ax.plot(tau, alpha * g_vals, 'g-', linewidth=2)
    if np.isfinite(window):
        ax.axvline(window, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Time lag τ (hours)')
    ax.set_ylabel('α·g̃(τ)')
    ax.set_title(f'Effective Temporal Kernel (α={alpha:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 4. Model summary and diagnostics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Load data to get event statistics
    with open(results['data_pickle'], 'rb') as f:
        data = pickle.load(f)
    events = data['events']
    t_events = np.array(events['t'])
    
    # Compute inter-event times
    inter_times = np.diff(np.sort(t_events))
    
    text = f"Model Summary:\n\n"
    text += f"Window W: {window:.2f} hours\n"
    text += f"Observation period T: {T:.1f} hours\n"
    text += f"Number of events: {len(t_events)}\n"
    text += f"Event rate: {len(t_events)/T:.2f} events/hour\n\n"
    
    text += f"Temporal kernel:\n"
    text += f"  - Normalized to unit integral\n"
    text += f"  - Window captures {window_mass:.1%} of mass\n"
    text += f"  - Mean lag (full): {np.sum(tau * g_vals) / np.sum(g_vals) * (tau[1]-tau[0]):.2f} hours\n"
    if np.isfinite(window):
        mask = tau <= window
        if np.sum(g_vals[mask]) > 0:
            mean_lag_window = np.sum(tau[mask] * g_vals[mask]) / np.sum(g_vals[mask]) * (tau[1]-tau[0])
            text += f"  - Mean lag (in window): {mean_lag_window:.2f} hours\n"
    
    text += f"\nInter-event times:\n"
    text += f"  - Mean: {np.mean(inter_times):.3f} hours\n"
    text += f"  - Median: {np.median(inter_times):.3f} hours\n"
    text += f"  - 95th percentile: {np.percentile(inter_times, 95):.3f} hours\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Temporal Kernel Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Diagnostic plots for Hawkes models')
    parser.add_argument('--result', type=str, required=True, help='Path to inference result pickle')
    args = parser.parse_args()
    
    results, state = load_results_and_state(args.result)
    plot_temporal_kernel_analysis(results, state)
    
    print("\nKey insights:")
    print("- The temporal kernel g̃ is normalized to have unit integral over [0,∞)")
    print("- With a finite window W, only a fraction of this mass is captured")
    print("- This is correctly handled in the model's likelihood calculation")
    print("- The issue with undercalibration suggests the model parameters may not be optimal")

if __name__ == '__main__':
    main() 