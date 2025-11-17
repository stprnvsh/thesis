"""
Analyze Non-Parametric HSGP Hawkes Results
Load and analyze the generated MCMC samples
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os
from datetime import datetime

def load_latest_results():
    """Load the latest non-parametric results"""
    # Find the most recent nonparametric results file
    pattern = "nonparametric_hawkes_hsgp_results_*.pickle"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No non-parametric results files found!")
        return None
    
    # Get the latest file
    latest_file = max(files, key=os.path.getctime)
    print(f"📂 Loading: {latest_file}")
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    return data, latest_file

def analyze_nonparametric_results(samples, true_values):
    """Analyze learned parameters vs true values for non-parametric model"""
    
    # Learned values (posterior means)
    # Reconstruct mark kernel from individual elements
    learned_mark_kernel = jnp.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            if f'mark_{i}_{j}' in samples:
                learned_mark_kernel = learned_mark_kernel.at[i, j].set(
                    float(jnp.mean(samples[f'mark_{i}_{j}']))
                )
    
    learned_values = {
        'mark_kernel': learned_mark_kernel,
        'spatial_amplitude': float(jnp.mean(samples['spatial/amplitude'])),
        'spatial_lengthscale': float(jnp.mean(samples['spatial/lengthscale'])),
        'temporal_amplitude': float(jnp.mean(samples['temporal/amplitude'])),
        'temporal_lengthscale': float(jnp.mean(samples['temporal/lengthscale'])),
    }
    
    print(f"\n🎯 NON-PARAMETRIC PARAMETER RECOVERY ANALYSIS:")
    print(f"=" * 70)
    
    print(f"\nLearned HSGP parameters:")
    print(f"  Spatial amplitude:   {learned_values['spatial_amplitude']:.3f}")
    print(f"  Spatial lengthscale: {learned_values['spatial_lengthscale']:.3f}")
    print(f"  Temporal amplitude:  {learned_values['temporal_amplitude']:.3f}")
    print(f"  Temporal lengthscale:{learned_values['temporal_lengthscale']:.3f}")
    
    print(f"\n📊 MARK KERNEL COMPARISON:")
    print(f"  True:\n{true_values['mark_kernel']}")
    print(f"  Learned:\n{learned_values['mark_kernel']}")
    
    # Mark kernel analysis
    true_mk = true_values['mark_kernel']
    learned_mk = learned_values['mark_kernel']
    mark_rmse = float(jnp.sqrt(jnp.mean((learned_mk - true_mk)**2)))
    
    print(f"\n  Mark RMSE: {mark_rmse:.4f}")
    
    # Element-wise analysis
    print(f"\n  Element-wise recovery:")
    for i in range(2):
        for j in range(2):
            true_val = true_mk[i, j]
            learned_val = learned_mk[i, j]
            error = abs(learned_val - true_val)
            error_pct = (error / true_val) * 100
            print(f"    [{i},{j}]: True={true_val:.3f}, Learned={learned_val:.3f}, Error={error:.3f} ({error_pct:.1f}%)")
    
    if mark_rmse < 0.15:
        print("🎉 EXCELLENT mark kernel recovery!")
    elif mark_rmse < 0.25:
        print("✅ Good mark kernel recovery")
    else:
        print("⚠️  Mark kernel recovery needs improvement")
        
    return {
        'mark_rmse': mark_rmse,
        'learned_values': learned_values
    }

def plot_hsgp_functions(samples, num_plot_samples=100):
    """Plot samples from the learned HSGP functions"""
    
    # Get some posterior samples
    n_samples = len(samples['spatial/amplitude'])
    indices = np.random.choice(n_samples, min(num_plot_samples, n_samples), replace=False)
    
    print(f"\n📈 Plotting HSGP function samples...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot spatial amplitude and lengthscale distributions
    ax1.hist(samples['spatial/amplitude'], bins=50, alpha=0.7, color='blue')
    ax1.set_title('Spatial Amplitude Distribution')
    ax1.set_xlabel('Amplitude')
    ax1.axvline(np.mean(samples['spatial/amplitude']), color='red', linestyle='--', label='Mean')
    ax1.legend()
    
    ax2.hist(samples['spatial/lengthscale'], bins=50, alpha=0.7, color='green')
    ax2.set_title('Spatial Lengthscale Distribution')
    ax2.set_xlabel('Lengthscale')
    ax2.axvline(np.mean(samples['spatial/lengthscale']), color='red', linestyle='--', label='Mean')
    ax2.legend()
    
    # Plot temporal amplitude and lengthscale distributions
    ax3.hist(samples['temporal/amplitude'], bins=50, alpha=0.7, color='purple')
    ax3.set_title('Temporal Amplitude Distribution')
    ax3.set_xlabel('Amplitude')
    ax3.axvline(np.mean(samples['temporal/amplitude']), color='red', linestyle='--', label='Mean')
    ax3.legend()
    
    ax4.hist(samples['temporal/lengthscale'], bins=50, alpha=0.7, color='orange')
    ax4.set_title('Temporal Lengthscale Distribution')
    ax4.set_xlabel('Lengthscale')
    ax4.axvline(np.mean(samples['temporal/lengthscale']), color='red', linestyle='--', label='Mean')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('nonparametric_hsgp_parameters.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ HSGP parameter plots saved: nonparametric_hsgp_parameters.png")

def compare_with_parametric():
    """Compare non-parametric results with parametric results if available"""
    
    # Try to load parametric results
    parametric_files = glob.glob("parametric_hawkes_results_*.pickle")
    
    if not parametric_files:
        print("⚠️  No parametric results found for comparison")
        return
    
    latest_parametric = max(parametric_files, key=os.path.getctime)
    print(f"\n🔄 Comparing with parametric results: {latest_parametric}")
    
    with open(latest_parametric, 'rb') as f:
        parametric_data = pickle.load(f)
    
    # Extract parametric learned values
    param_analysis = parametric_data['analysis_results']
    
    print(f"\n📊 PARAMETRIC vs NON-PARAMETRIC COMPARISON:")
    print(f"=" * 60)
    print(f"Parameter Recovery (Mark Kernel RMSE):")
    print(f"  Parametric:     {param_analysis['mark_rmse']:.4f}")
    
    # We'll get non-parametric RMSE from the main analysis
    return parametric_data

def main():
    """Main analysis function"""
    print("=" * 80)
    print("🔍 ANALYZING NON-PARAMETRIC HSGP HAWKES RESULTS")
    print("=" * 80)
    
    # Load results
    data, filename = load_latest_results()
    if data is None:
        return
    
    samples = data['mcmc_samples']
    true_values = data['true_parameters']
    
    print(f"\n📂 Loaded results from: {filename}")
    print(f"   Model type: {data['model_type']}")
    print(f"   Timestamp: {data['timestamp']}")
    print(f"   Number of samples: {len(samples['spatial/amplitude'])}")
    print(f"   HSGP config: {data['hsgp_config']}")
    
    # Analyze results
    results = analyze_nonparametric_results(samples, true_values)
    
    # Plot HSGP parameters
    plot_hsgp_functions(samples)
    
    # Compare with parametric if available
    parametric_data = compare_with_parametric()
    
    if parametric_data:
        param_rmse = parametric_data['analysis_results']['mark_rmse']
        nonparam_rmse = results['mark_rmse']
        
        print(f"  Non-parametric: {nonparam_rmse:.4f}")
        print(f"\n🎯 Comparison:")
        if nonparam_rmse < param_rmse:
            print(f"   ✅ Non-parametric BETTER by {param_rmse - nonparam_rmse:.4f}")
        elif nonparam_rmse > param_rmse:
            print(f"   ⚠️  Parametric BETTER by {nonparam_rmse - param_rmse:.4f}")
        else:
            print(f"   🟰 Both methods perform similarly")
    
    # Print summary
    print(f"\n" + "=" * 80)
    print("🎉 NON-PARAMETRIC HSGP ANALYSIS COMPLETE!")
    print(f"✅ Flexible spatial kernel learned via HSGP")
    print(f"✅ Flexible temporal kernel learned via HSGP")
    print(f"✅ Mark kernel RMSE: {results['mark_rmse']:.4f}")
    print(f"💾 Plots saved: nonparametric_hsgp_parameters.png")
    print("=" * 80)

if __name__ == "__main__":
    main() 