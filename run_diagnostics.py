#!/usr/bin/env python3
"""
Run Bayesian diagnostics on specific Hawkes inference results
"""

import subprocess
import sys
from pathlib import Path

def run_diagnostics():
    # Define the files to analyze
    files_to_analyze = [
        {
            "name": "np3_test_arbon_events_evening",
            "state_file": "mcmc_state_np3_test_arbon_events_evening.npz",
            "result_file": "inference_result_np3_test_arbon_events_evening.pickle"
        },
        {
            "name": "np_large_arbon_events_evening_copy_linear", 
            "state_file": "mcmc_state_np_large_arbon_events_evening_copy_linear.npz",
            "result_file": "inference_result_np_large_arbon_events_evening_copy_linear.pickle"
        },
        {
            "name": "np_large_arbon_events_evening_copy_relu",
            "state_file": "mcmc_state_np_large_arbon_events_evening_copy.npz",  # Note: this is the base file
            "result_file": "inference_result_np_large_arbon_events_evening_copy_relu.pickle"
        },
        {
            "name": "np_large_arbon_events_evening_copy_softplus",
            "state_file": "mcmc_state_np_large_arbon_events_evening_copy.npz",  # Note: this is the base file
            "result_file": "inference_result_np_large_arbon_events_evening_copy_softplus.pickle"
        }
    ]
    
    for file_info in files_to_analyze:
        print(f"\n{'='*60}")
        print(f"Running diagnostics for: {file_info['name']}")
        print(f"{'='*60}")
        
        # Check if state file exists
        state_path = Path(file_info['state_file'])
        if not state_path.exists():
            print(f"⚠️  State file not found: {state_path}")
            print("   This means you only have posterior means, not full MCMC samples.")
            print("   For proper Bayesian diagnostics, you need the .npz state files.")
            continue
        
        # Check if result file exists
        result_path = Path(file_info['result_file'])
        if not result_path.exists():
            print(f"⚠️  Result file not found: {result_path}")
            continue
        
        # Create output directory
        output_dir = f"diagnostics_{file_info['name']}"
        
        # Run diagnostics
        cmd = [
            sys.executable, "bayesian_diagnostics.py",
            "--state_file", file_info['state_file'],
            "--result_file", file_info['result_file'],
            "--save_dir", output_dir,
            "--plot_traces"
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Diagnostics completed successfully")
                print(f"Results saved to: {output_dir}/")
            else:
                print("❌ Diagnostics failed")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                
        except subprocess.TimeoutExpired:
            print("⏰ Diagnostics timed out after 5 minutes")
        except Exception as e:
            print(f"❌ Error running diagnostics: {e}")

if __name__ == "__main__":
    run_diagnostics() 