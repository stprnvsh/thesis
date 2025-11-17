#!/usr/bin/env python3
"""
Test script for improved kernel plotting
"""

import numpy as np
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt

def load_hsgp_results(filename):
    """Load HSGP results"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_improved_kernels(samples, save_prefix="improved_kernels"):
    """
    Improved kernel plotting with correct spatial kernel normalization
    """
    print(f"\n📊 PLOTTING IMPROVED KERNELS...")
    print(f"   NOTE: Fixed spatial kernel normalization and parameterization")
    
    # Extract HSGP samples
    spatial_amplitude = samples['spatial_amplitude']
    spatial_lengthscale = samples['spatial_lengthscale']
    temporal_amplitude = samples['temporal_amplitude']
    temporal_lengthscale = samples['temporal_lengthscale']
    
    # Get posterior means
    spatial_amp_mean = float(jnp.mean(spatial_amplitude))
    spatial_len_mean = float(jnp.mean(spatial_lengthscale))
    temporal_amp_mean = float(jnp.mean(temporal_amplitude))
    temporal_len_mean = float(jnp.mean(temporal_lengthscale))
    
    print(f"   Spatial HSGP: amplitude={spatial_amp_mean:.3f}, lengthscale={spatial_len_mean:.3f}")
    print(f"   Temporal HSGP: amplitude={temporal_amp_mean:.3f}, lengthscale={temporal_len_mean:.3f}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CORRECTED: Learned Non-Parametric Kernels vs True Parametric Kernels', fontsize=16)
    
    # === SPATIAL KERNEL COMPARISON (FIXED) ===
    distances = jnp.linspace(0, 10, 100)
    
    # True parametric spatial kernel: Gaussian shape (removed tiny normalization)
    true_sigma = 2.0
    true_spatial_kernel = jnp.exp(-distances**2 / (2 * true_sigma**2))
    
    # Learned spatial kernel approximation using HSGP hyperparameters
    learned_spatial_approx = spatial_amp_mean * jnp.exp(-distances**2 / (2 * spatial_len_mean**2))
    
    axes[0, 0].plot(distances, true_spatial_kernel, 'b-', linewidth=2, label='True Parametric (Gaussian)')
    axes[0, 0].plot(distances, learned_spatial_approx, 'r--', linewidth=2, label='Learned HSGP (approx)')
    axes[0, 0].set_xlabel('Spatial Distance')
    axes[0, 0].set_ylabel('Kernel Value (unnormalized)')
    axes[0, 0].set_title('FIXED: Spatial Kernel Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # === TEMPORAL KERNEL COMPARISON ===
    delta_t = jnp.linspace(0.1, 8, 100)
    
    # True parametric temporal kernel: ω * exp(-ω * Δt), ω=1.0
    true_omega = 1.0
    true_temporal_kernel = true_omega * jnp.exp(-true_omega * delta_t)
    
    # Learned temporal kernel approximation
    learned_temporal_approx = temporal_amp_mean * jnp.exp(-delta_t / temporal_len_mean)
    
    axes[0, 1].plot(delta_t, true_temporal_kernel, 'b-', linewidth=2, label='True Parametric (Exponential)')
    axes[0, 1].plot(delta_t, learned_temporal_approx, 'r--', linewidth=2, label='Learned HSGP (approx)')
    axes[0, 1].set_xlabel('Time Difference (Δt)')
    axes[0, 1].set_ylabel('Kernel Value')
    axes[0, 1].set_title('Temporal Kernel Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # === MARK KERNEL COMPARISON ===
    learned_mark_kernel = jnp.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            if f'mark_{i}_{j}' in samples:
                learned_mark_kernel = learned_mark_kernel.at[i, j].set(
                    float(jnp.mean(samples[f'mark_{i}_{j}']))
                )
    
    # True mark kernel
    true_mark_kernel = jnp.array([[0.8, 0.4], [0.3, 0.9]])
    
    # Plot mark kernels as heatmaps
    im1 = axes[1, 0].imshow(true_mark_kernel, cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title('True Mark Kernel')
    axes[1, 0].set_xlabel('Source Event Type')
    axes[1, 0].set_ylabel('Destination Event Type')
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f'{true_mark_kernel[i, j]:.2f}', 
                          ha='center', va='center', fontweight='bold')
    
    im2 = axes[1, 1].imshow(learned_mark_kernel, cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title('Learned Mark Kernel')
    axes[1, 1].set_xlabel('Source Event Type')
    axes[1, 1].set_ylabel('Destination Event Type')
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, f'{learned_mark_kernel[i, j]:.2f}', 
                          ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{save_prefix}_corrected.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   💾 CORRECTED kernel plot saved: {filename}")
    
    plt.show()
    
    print(f"   📊 Kernel comparison:")
    print(f"      True spatial (Gaussian): σ={true_sigma}")
    print(f"      Learned spatial: amp={spatial_amp_mean:.3f}, len={spatial_len_mean:.3f}")
    print(f"      True temporal (Exponential): ω={true_omega}")
    print(f"      Learned temporal: amp={temporal_amp_mean:.3f}, len={temporal_len_mean:.3f}")
    
    return {
        'spatial_amp_mean': spatial_amp_mean,
        'spatial_len_mean': spatial_len_mean,
        'temporal_amp_mean': temporal_amp_mean,
        'temporal_len_mean': temporal_len_mean,
        'learned_mark_kernel': learned_mark_kernel
    }

def main():
    """Test the improved plotting"""
    print("Testing improved kernel plotting...")
    
    # Load the most recent HSGP results
    filename = "nonparametric_hawkes_hsgp_results_20250804_070414.pickle"
    
    try:
        print(f"Loading {filename}...")
        data = load_hsgp_results(filename)
        
        samples = data['mcmc_samples']
        print(f"Found {len(samples)} parameter samples")
        
        # Test improved plotting
        kernel_analysis = plot_improved_kernels(samples)
        
        print(f"\n✅ Improved plotting test complete!")
        print(f"Key improvements:")
        print(f"  🔧 Fixed spatial kernel normalization (now visible)")
        print(f"  🔧 Added proper labels (Gaussian vs Exponential)")
        print(f"  🔧 Consistent parameterizations")
        print(f"  🔧 Better approximation explanations")
        
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 