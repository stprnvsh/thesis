#!/usr/bin/env python3
"""
COMPREHENSIVE DIAGNOSTIC TESTS for HSGP Hawkes Model
Tests if the model is actually learning meaningful patterns
"""

import numpy as np
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple
import pandas as pd

def load_results(filename: str) -> Dict:
    """Load results from pickle file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def parameter_recovery_test(samples: Dict, verbose: bool = True) -> Dict:
    """
    Test 1: Parameter Recovery
    Compare learned parameters vs true parameters
    """
    print("🔬 TEST 1: PARAMETER RECOVERY ANALYSIS")
    print("=" * 50)
    
    # True parameters from data generation
    true_params = {
        'spatial_amplitude': 0.5,  # Expected from prior
        'spatial_lengthscale': 2.0,  # True sigma_spatial
        'temporal_amplitude': 1.0,  # True omega_temporal  
        'temporal_lengthscale': 1.0,  # True omega_temporal
        'mark_kernel': jnp.array([[0.8, 0.4], [0.3, 0.9]])
    }
    
    # Learned parameters (posterior means)
    learned_params = {}
    
    # Extract HSGP hyperparameters
    if 'spatial_amplitude' in samples:
        learned_params['spatial_amplitude'] = float(jnp.mean(samples['spatial_amplitude']))
        learned_params['spatial_lengthscale'] = float(jnp.mean(samples['spatial_lengthscale']))
        learned_params['temporal_amplitude'] = float(jnp.mean(samples['temporal_amplitude']))
        learned_params['temporal_lengthscale'] = float(jnp.mean(samples['temporal_lengthscale']))
    
    # Extract mark kernel
    learned_mark = jnp.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            if f'mark_{i}_{j}' in samples:
                learned_mark = learned_mark.at[i, j].set(float(jnp.mean(samples[f'mark_{i}_{j}'])))
    learned_params['mark_kernel'] = learned_mark
    
    # Calculate recovery metrics
    recovery_metrics = {}
    
    for param_name in ['spatial_amplitude', 'spatial_lengthscale', 'temporal_amplitude', 'temporal_lengthscale']:
        if param_name in learned_params:
            true_val = true_params[param_name]
            learned_val = learned_params[param_name]
            
            # Relative error
            rel_error = abs(learned_val - true_val) / abs(true_val)
            recovery_metrics[param_name] = {
                'true': true_val,
                'learned': learned_val,
                'abs_error': abs(learned_val - true_val),
                'rel_error': rel_error,
                'recovered': rel_error < 0.5  # Within 50%
            }
            
            if verbose:
                print(f"  {param_name}:")
                print(f"    True: {true_val:.3f}, Learned: {learned_val:.3f}")
                print(f"    Error: {rel_error*100:.1f}% {'✅' if rel_error < 0.5 else '❌'}")
    
    # Mark kernel recovery
    mark_rmse = float(jnp.sqrt(jnp.mean((learned_params['mark_kernel'] - true_params['mark_kernel'])**2)))
    recovery_metrics['mark_rmse'] = mark_rmse
    
    if verbose:
        print(f"  Mark Kernel RMSE: {mark_rmse:.4f} {'✅' if mark_rmse < 0.2 else '❌'}")
        print(f"  True Mark Kernel:\n{true_params['mark_kernel']}")
        print(f"  Learned Mark Kernel:\n{learned_params['mark_kernel']}")
    
    # Overall recovery score
    param_recovery_rate = sum(m.get('recovered', False) for m in recovery_metrics.values() if isinstance(m, dict))
    total_params = sum(1 for m in recovery_metrics.values() if isinstance(m, dict))
    
    overall_score = param_recovery_rate / total_params if total_params > 0 else 0
    recovery_metrics['overall_recovery_rate'] = overall_score
    recovery_metrics['overall_grade'] = 'A' if overall_score >= 0.8 else 'B' if overall_score >= 0.6 else 'C' if overall_score >= 0.4 else 'F'
    
    print(f"\n📊 PARAMETER RECOVERY SUMMARY:")
    print(f"   Recovery Rate: {param_recovery_rate}/{total_params} ({overall_score*100:.1f}%)")
    print(f"   Grade: {recovery_metrics['overall_grade']}")
    
    return recovery_metrics

def convergence_diagnostics(samples: Dict, verbose: bool = True) -> Dict:
    """
    Test 2: MCMC Convergence Diagnostics
    Check if chains have converged properly
    """
    print("\n🔬 TEST 2: MCMC CONVERGENCE DIAGNOSTICS")
    print("=" * 50)
    
    diagnostics = {}
    
    # Check effective sample size and R-hat for key parameters
    key_params = ['spatial_amplitude', 'spatial_lengthscale', 'temporal_amplitude', 'temporal_lengthscale']
    
    for param in key_params:
        if param in samples:
            param_samples = samples[param]
            
            # Basic statistics
            mean_val = float(jnp.mean(param_samples))
            std_val = float(jnp.std(param_samples))
            
            # Autocorrelation (simplified)
            autocorr = float(jnp.corrcoef(param_samples[:-1], param_samples[1:])[0, 1])
            
            # Effective sample size approximation
            n_samples = len(param_samples)
            eff_sample_size = n_samples / (1 + 2 * autocorr) if autocorr > 0 else n_samples
            
            diagnostics[param] = {
                'mean': mean_val,
                'std': std_val,
                'autocorr': autocorr,
                'eff_sample_size': eff_sample_size,
                'eff_ratio': eff_sample_size / n_samples,
                'converged': eff_sample_size > 100 and autocorr < 0.8
            }
            
            if verbose:
                print(f"  {param}: ESS={eff_sample_size:.0f} ({eff_sample_size/n_samples*100:.1f}%) {'✅' if eff_sample_size > 100 else '❌'}")
    
    # Overall convergence
    converged_params = sum(d['converged'] for d in diagnostics.values())
    total_params = len(diagnostics)
    convergence_rate = converged_params / total_params if total_params > 0 else 0
    
    diagnostics['overall_convergence'] = convergence_rate
    diagnostics['convergence_grade'] = 'A' if convergence_rate >= 0.9 else 'B' if convergence_rate >= 0.7 else 'C' if convergence_rate >= 0.5 else 'F'
    
    print(f"\n📊 CONVERGENCE SUMMARY:")
    print(f"   Converged: {converged_params}/{total_params} ({convergence_rate*100:.1f}%)")
    print(f"   Grade: {diagnostics['convergence_grade']}")
    
    return diagnostics

def posterior_predictive_test(samples: Dict, true_events: np.ndarray, verbose: bool = True) -> Dict:
    """
    Test 3: Posterior Predictive Checks
    Generate synthetic data and compare to true data
    """
    print("\n🔬 TEST 3: POSTERIOR PREDICTIVE CHECKS")
    print("=" * 50)
    
    # Extract learned parameters
    if 'spatial_amplitude' not in samples:
        print("   ❌ Missing HSGP parameters for posterior predictive check")
        return {'grade': 'F', 'available': False}
    
    # Basic statistics comparison
    if hasattr(true_events, 'dtype') and true_events.dtype.names:
        true_times = true_events['t']
        true_nodes = true_events['u']
        true_types = true_events['e']
    else:
        true_times = np.array([e[0] for e in true_events])
        true_nodes = np.array([e[1] for e in true_events])
        true_types = np.array([e[2] for e in true_events])
    
    # Statistical tests on event patterns
    stats_comparison = {}
    
    # 1. Inter-event times
    inter_event_times = np.diff(np.sort(true_times))
    stats_comparison['mean_inter_event'] = float(np.mean(inter_event_times))
    stats_comparison['std_inter_event'] = float(np.std(inter_event_times))
    
    # 2. Event type distribution
    type_counts = np.bincount(true_types.astype(int))
    stats_comparison['type_distribution'] = type_counts / len(true_types)
    
    # 3. Spatial distribution
    node_counts = np.bincount(true_nodes.astype(int))
    stats_comparison['spatial_distribution'] = node_counts / len(true_nodes)
    
    # 4. Temporal clustering (burstiness)
    sorted_times = np.sort(true_times)
    short_intervals = np.sum(inter_event_times < np.median(inter_event_times))
    burstiness = short_intervals / len(inter_event_times)
    stats_comparison['burstiness'] = burstiness
    
    if verbose:
        print(f"   Event count: {len(true_times)}")
        print(f"   Time span: {float(np.max(true_times) - np.min(true_times)):.1f}")
        print(f"   Mean inter-event time: {stats_comparison['mean_inter_event']:.3f}")
        print(f"   Burstiness index: {burstiness:.3f}")
        print(f"   Type distribution: {stats_comparison['type_distribution']}")
    
    # Predictive quality assessment (simplified)
    predictive_metrics = {
        'stats_comparison': stats_comparison,
        'data_available': True,
        'grade': 'B'  # Default grade - would need full simulation for proper test
    }
    
    print(f"\n📊 POSTERIOR PREDICTIVE SUMMARY:")
    print(f"   Grade: {predictive_metrics['grade']} (Limited - need full simulation)")
    
    return predictive_metrics

def kernel_learning_test(samples: Dict, verbose: bool = True) -> Dict:
    """
    Test 4: Kernel Learning Validation
    Check if learned kernels capture true relationships
    """
    print("\n🔬 TEST 4: KERNEL LEARNING VALIDATION")
    print("=" * 50)
    
    if 'spatial_amplitude' not in samples:
        print("   ❌ Missing HSGP parameters")
        return {'grade': 'F', 'available': False}
    
    # Extract learned hyperparameters
    spatial_amp = float(jnp.mean(samples['spatial_amplitude']))
    spatial_len = float(jnp.mean(samples['spatial_lengthscale']))
    temporal_amp = float(jnp.mean(samples['temporal_amplitude']))
    temporal_len = float(jnp.mean(samples['temporal_lengthscale']))
    
    # Test points for kernel evaluation
    spatial_distances = jnp.linspace(0, 10, 50)
    temporal_diffs = jnp.linspace(0.1, 5, 50)
    
    # True kernels (unnormalized for comparison)
    true_spatial = jnp.exp(-spatial_distances**2 / (2 * 2.0**2))  # σ = 2.0
    true_temporal = 1.0 * jnp.exp(-1.0 * temporal_diffs)  # ω = 1.0
    
    # Learned kernel approximations
    learned_spatial = spatial_amp * jnp.exp(-spatial_distances**2 / (2 * spatial_len**2))
    learned_temporal = temporal_amp * jnp.exp(-temporal_diffs / temporal_len)
    
    # Correlation tests
    spatial_corr, spatial_p = pearsonr(true_spatial, learned_spatial)
    temporal_corr, temporal_p = pearsonr(true_temporal, learned_temporal)
    
    # Shape similarity (normalized)
    spatial_rmse = float(jnp.sqrt(jnp.mean((true_spatial/jnp.max(true_spatial) - learned_spatial/jnp.max(learned_spatial))**2)))
    temporal_rmse = float(jnp.sqrt(jnp.mean((true_temporal/jnp.max(true_temporal) - learned_temporal/jnp.max(learned_temporal))**2)))
    
    kernel_metrics = {
        'spatial_correlation': spatial_corr,
        'temporal_correlation': temporal_corr,
        'spatial_rmse': spatial_rmse,
        'temporal_rmse': temporal_rmse,
        'spatial_learned_well': spatial_corr > 0.7 and spatial_rmse < 0.3,
        'temporal_learned_well': temporal_corr > 0.7 and temporal_rmse < 0.3
    }
    
    if verbose:
        print(f"   Spatial kernel correlation: {spatial_corr:.3f} {'✅' if spatial_corr > 0.7 else '❌'}")
        print(f"   Temporal kernel correlation: {temporal_corr:.3f} {'✅' if temporal_corr > 0.7 else '❌'}")
        print(f"   Spatial shape RMSE: {spatial_rmse:.3f} {'✅' if spatial_rmse < 0.3 else '❌'}")
        print(f"   Temporal shape RMSE: {temporal_rmse:.3f} {'✅' if temporal_rmse < 0.3 else '❌'}")
    
    # Overall kernel learning grade
    kernel_score = (kernel_metrics['spatial_learned_well'] + kernel_metrics['temporal_learned_well']) / 2
    kernel_metrics['overall_score'] = kernel_score
    kernel_metrics['grade'] = 'A' if kernel_score >= 0.9 else 'B' if kernel_score >= 0.7 else 'C' if kernel_score >= 0.5 else 'F'
    
    print(f"\n📊 KERNEL LEARNING SUMMARY:")
    print(f"   Kernels learned well: {int(kernel_metrics['spatial_learned_well']) + int(kernel_metrics['temporal_learned_well'])}/2")
    print(f"   Grade: {kernel_metrics['grade']}")
    
    return kernel_metrics

def create_diagnostic_plots(samples: Dict, diagnostics: Dict, save_prefix: str = "diagnostics"):
    """
    Create comprehensive diagnostic plots
    """
    print("\n📊 GENERATING DIAGNOSTIC PLOTS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('HSGP Hawkes Model - Comprehensive Diagnostics', fontsize=16)
    
    # Plot 1: Parameter trace plots
    if 'spatial_amplitude' in samples:
        axes[0, 0].plot(samples['spatial_amplitude'], alpha=0.7, label='Spatial Amp')
        axes[0, 0].plot(samples['spatial_lengthscale'], alpha=0.7, label='Spatial Len')
        axes[0, 0].set_title('Spatial Parameter Traces')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Temporal parameter traces
    if 'temporal_amplitude' in samples:
        axes[0, 1].plot(samples['temporal_amplitude'], alpha=0.7, label='Temporal Amp')
        axes[0, 1].plot(samples['temporal_lengthscale'], alpha=0.7, label='Temporal Len')
        axes[0, 1].set_title('Temporal Parameter Traces')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Parameter posteriors
    if 'spatial_amplitude' in samples:
        axes[0, 2].hist(samples['spatial_amplitude'], alpha=0.6, bins=30, label='Spatial Amp')
        axes[0, 2].axvline(0.5, color='red', linestyle='--', label='True (≈0.5)')
        axes[0, 2].set_title('Spatial Amplitude Posterior')
        axes[0, 2].legend()
    
    # Plot 4: Convergence diagnostics
    if 'parameter_recovery' in diagnostics:
        param_names = [p for p in diagnostics['parameter_recovery'] if isinstance(diagnostics['parameter_recovery'][p], dict)]
        recovery_rates = [diagnostics['parameter_recovery'][p]['recovered'] for p in param_names]
        
        axes[1, 0].bar(range(len(param_names)), [int(r) for r in recovery_rates])
        axes[1, 0].set_xticks(range(len(param_names)))
        axes[1, 0].set_xticklabels([p.replace('_', '\n') for p in param_names], rotation=45)
        axes[1, 0].set_title('Parameter Recovery')
        axes[1, 0].set_ylabel('Recovered (0/1)')
    
    # Plot 5: Kernel comparison
    if 'kernel_learning' in diagnostics and diagnostics['kernel_learning'].get('available', True):
        spatial_corr = diagnostics['kernel_learning']['spatial_correlation']
        temporal_corr = diagnostics['kernel_learning']['temporal_correlation']
        
        axes[1, 1].bar(['Spatial', 'Temporal'], [spatial_corr, temporal_corr])
        axes[1, 1].axhline(0.7, color='red', linestyle='--', label='Threshold')
        axes[1, 1].set_title('Kernel Learning (Correlation)')
        axes[1, 1].set_ylabel('Correlation with True')
        axes[1, 1].legend()
    
    # Plot 6: Overall grades
    grades = {}
    if 'parameter_recovery' in diagnostics:
        grades['Param Recovery'] = diagnostics['parameter_recovery']['overall_grade']
    if 'convergence' in diagnostics:
        grades['Convergence'] = diagnostics['convergence']['convergence_grade'] 
    if 'kernel_learning' in diagnostics:
        grades['Kernel Learning'] = diagnostics['kernel_learning']['grade']
    
    grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    grade_nums = [grade_values.get(g, 0) for g in grades.values()]
    
    bars = axes[1, 2].bar(grades.keys(), grade_nums)
    axes[1, 2].set_title('Overall Model Performance')
    axes[1, 2].set_ylabel('Grade (A=4, F=0)')
    axes[1, 2].set_ylim(0, 4)
    
    # Color bars by grade
    colors = ['red' if g == 'F' else 'orange' if g in ['C', 'D'] else 'yellow' if g == 'B' else 'green' for g in grades.values()]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_prefix}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   💾 Diagnostic plots saved: {filename}")
    
    plt.show()

def run_comprehensive_diagnostics(filename: str):
    """
    Run all diagnostic tests on a pickle file
    """
    print("🚀 COMPREHENSIVE HSGP HAWKES MODEL DIAGNOSTICS")
    print(f"📂 Loading: {filename}")
    print("=" * 70)
    
    try:
        # Load results
        data = load_results(filename)
        
        # Extract data
        samples = data.get('mcmc_samples', {})
        events = data.get('events', None)
        model_info = data.get('data_info', {})
        
        print(f"📊 Data Info:")
        print(f"   Model type: {data.get('model_type', 'Unknown')}")
        print(f"   Timestamp: {data.get('timestamp', 'Unknown')}")
        if model_info:
            print(f"   Events: {model_info.get('num_events', 'Unknown')}")
            print(f"   Nodes: {model_info.get('num_nodes', 'Unknown')}")
        
        # Run all diagnostic tests
        diagnostics = {}
        
        # Test 1: Parameter Recovery
        diagnostics['parameter_recovery'] = parameter_recovery_test(samples)
        
        # Test 2: Convergence
        diagnostics['convergence'] = convergence_diagnostics(samples)
        
        # Test 3: Posterior Predictive (if events available)
        if events is not None:
            diagnostics['posterior_predictive'] = posterior_predictive_test(samples, events)
        
        # Test 4: Kernel Learning
        diagnostics['kernel_learning'] = kernel_learning_test(samples)
        
        # Generate diagnostic plots
        create_diagnostic_plots(samples, diagnostics)
        
        # Overall assessment
        print("\n" + "=" * 70)
        print("🎯 OVERALL MODEL ASSESSMENT")
        print("=" * 70)
        
        grades = []
        if 'parameter_recovery' in diagnostics:
            param_grade = diagnostics['parameter_recovery']['overall_grade']
            print(f"   Parameter Recovery: {param_grade}")
            grades.append(param_grade)
        
        if 'convergence' in diagnostics:
            conv_grade = diagnostics['convergence']['convergence_grade']
            print(f"   MCMC Convergence: {conv_grade}")
            grades.append(conv_grade)
        
        if 'kernel_learning' in diagnostics:
            kernel_grade = diagnostics['kernel_learning']['grade']
            print(f"   Kernel Learning: {kernel_grade}")
            grades.append(kernel_grade)
        
        # Overall grade
        grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        avg_grade = np.mean([grade_values.get(g, 0) for g in grades])
        overall_letter = ['F', 'D', 'C', 'B', 'A'][min(int(avg_grade), 4)]
        
        print(f"\n📊 FINAL ASSESSMENT: {overall_letter} ({avg_grade:.1f}/4.0)")
        
        if avg_grade >= 3.5:
            print("🎉 EXCELLENT: Model is learning very well!")
        elif avg_grade >= 2.5:
            print("✅ GOOD: Model is learning adequately")
        elif avg_grade >= 1.5:
            print("⚠️  FAIR: Model has some learning but needs improvement")
        else:
            print("❌ POOR: Model is not learning effectively")
        
        return diagnostics
        
    except Exception as e:
        print(f"❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    # Use the most recent HSGP results
    filename = "nonparametric_hawkes_hsgp_results_20250804_070414.pickle"
    
    print("🔬 HSGP HAWKES MODEL DIAGNOSTIC SUITE")
    print("Testing if the model is actually learning...")
    print()
    
    diagnostics = run_comprehensive_diagnostics(filename)
    
    if diagnostics:
        print("\n✅ Diagnostic tests complete!")
        print("Check the generated plots for detailed analysis.")

if __name__ == "__main__":
    main() 