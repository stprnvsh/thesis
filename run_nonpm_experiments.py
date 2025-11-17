#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch script to run nonpm_window experiments with different nonlinearity options in parallel.
"""

import subprocess
import sys
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"🚀 Starting: {description}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Command: {' '.join(cmd)}")
    print(f"🆔 Process ID: {os.getpid()}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ SUCCESS: {description}")
        print(f"⏱️  Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        return True, description, duration
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description}")
        print(f"Error code: {e.returncode}")
        return False, description, 0
    except KeyboardInterrupt:
        print(f"\n⚠️  INTERRUPTED: {description}")
        return False, description, 0

def run_experiment(experiment_config):
    """Run a single experiment."""
    script, cmd, description = experiment_config
    return run_command(cmd, description)

def main():
    # Base command parameters for nonpm_window_4.py
    base_cmd_4 = [
        "python", "nonpm_window_4.py",
        "--data", "large_arbon_events_evening_copy.pickle",
        "--method", "mcmc",
        "--warmup", "3000",
        "--samples", "3000", 
        "--chains", "1",
        "--B_t", "20",
        "--B_s", "20",
        "--window", "1"
    ]
    
    # Base command parameters for nonpm_window_3.py
    base_cmd_3 = [
        "python", "nonpm_window_3.py",
        "--data", "large_arbon_events_evening_copy.pickle",
        "--method", "mcmc",
        "--warmup", "3000",
        "--samples", "3000", 
        "--chains", "1",
        "--B_t", "20",
        "--B_r", "20",
        "--window", "1"
    ]
    
    # All experiments configuration
    experiments = [
        # nonpm_window_4.py experiments
        #("nonpm_window_4.py", base_cmd_4 + ["--data", "large_arbon_events_evening_copy.pickle","--nonlinearity", "linear"], "Linear Hawkes Model (nonpm4)"),
        #("nonpm_window_4.py", base_cmd_4 + ["--data", "large_arbon_events_evening_copy.pickle","--nonlinearity", "softplus"], "Softplus Nonlinear Hawkes Model (nonpm4)"),
        #("nonpm_window_4.py", base_cmd_4 + ["--data", "large_arbon_events_evening_copy.pickle","--nonlinearity", "relu"], "ReLU Nonlinear Hawkes Model (nonpm4)"), 
        ("nonpm_window_4.py", base_cmd_4 + ["--nonlinearity", "exp"], "Exp Nonlinear Hawkes Model (nonpm4)"),
        ("nonpm_window_4.py", base_cmd_4 + ["--nonlinearity", "power2"], "Power2 Nonlinear Hawkes Model (nonpm4)"),
        ("nonpm_window_4.py", base_cmd_4 + ["--nonlinearity", "linear", "--use_qhp", "--B_q", "10", "--q_scale", "0.5"], "Linear QHP Nonlinear Hawkes Model (nonpm4)"),
        #("nonpm_window_4.py", base_cmd_4 + ["--nonlinearity", "relu", "--use_qhp", "--B_q", "10", "--q_scale", "0.5"], "ReLU QHP Nonlinear Hawkes Model (nonpm4)"),
        #("nonpm_window_4.py", base_cmd_4 + ["--nonlinearity", "softplus", "--use_qhp", "--B_q", "10", "--q_scale", "0.5"], "Softplus QHP Nonlinear Hawkes Model (nonpm4)"),
        #("nonpm_window_4.py", base_cmd_4 + ["--nonlinearity", "exp", "--use_qhp", "--B_q", "10", "--q_scale", "0.5"], "Exp QHP Nonlinear Hawkes Model (nonpm4)"),
        #("nonpm_window_4.py", base_cmd_4 + ["--nonlinearity", "power2", "--use_qhp", "--B_q", "10", "--q_scale", "0.5"], "Power2 QHP Nonlinear Hawkes Model (nonpm4)"),
        # nonpm_window_3.py experiment (joint spatio-temporal, linear only)
        #("nonpm_window_3.py", base_cmd_3, "Joint Spatio-Temporal Hawkes Model (nonpm3)")
    ]
    
    print("🎯 Starting Nonparametric Hawkes Experiments (Parallel)")
    print(f"📊 Data: lage_arbon_events_morning.pickle")
    print(f"🔬 Method: MCMC (3000 warmup, 3000 samples, 1 chain per experiment)")
    print(f"📐 Basis: B_t=20, B_s/B_r=20, window=0.5")
    print(f"🧪 Models: linear, relu, softmax (nonpm4) + joint spatio-temporal (nonpm3)")
    print(f"⚡ Running {len(experiments)} experiments in parallel")
    
    # Determine number of parallel processes
    max_workers = min(len(experiments), mp.cpu_count())
    print(f"🔧 Using {max_workers} parallel processes")
    
    successful_runs = 0
    total_runs = len(experiments)
    results = []
    
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(run_experiment, exp_config): exp_config 
            for exp_config in experiments
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_exp):
            exp_config = future_to_exp[future]
            try:
                success, description, duration = future.result()
                results.append((success, description, duration))
                if success:
                    successful_runs += 1
                print(f"📊 Completed: {description} - {'✅ SUCCESS' if success else '❌ FAILED'}")
            except Exception as e:
                print(f"❌ Exception in {exp_config[2]}: {e}")
                results.append((False, exp_config[2], 0))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful runs: {successful_runs}/{total_runs}")
    print(f"❌ Failed runs: {total_runs - successful_runs}/{total_runs}")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for success, description, duration in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        time_str = f"{duration:.1f} min" if duration > 0 else "N/A"
        print(f"  {status} | {description} | {time_str}")
    
    if successful_runs == total_runs:
        print(f"\n🎉 All experiments completed successfully!")
    else:
        print(f"\n⚠️  Some experiments failed. Check the output above.")
    
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 