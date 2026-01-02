# modules/utils.py
# Utility functions for plotting learning curves and DORAEMON dynamics.

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

def plot_learning_curve(log_dir, title="Learning Curve"):
    """
    Reads all monitor files in log_dir and plots smoothed rewards.
    Robust to corrupted lines and different naming conventions.
    """
    # 1. Find files (support both naming conventions)
    files = glob.glob(f"{log_dir}/*.monitor.csv")
    if not files:
        files = glob.glob(f"{log_dir}/monitor.csv")
        
    if not files:
        print(f"No monitor files found in {log_dir}")
        return

    print(f"Found {len(files)} log files: {[os.path.basename(f) for f in files]}")

    data_frames = []
    for file in files:
        try:
            # skiprows=1 handles the JSON header
            # on_bad_lines='skip' ignores corrupted lines (e.g. from crashes)
            df = pd.read_csv(file, skiprows=1, on_bad_lines='skip')
            data_frames.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")
            continue
            
    if not data_frames:
        print("No valid data found to plot.")
        return

    # 2. Combine and Sort
    df_concat = pd.concat(data_frames)
    df_concat = df_concat.sort_values(by='t') # Sort by walltime
    
    # 3. Smoothing
    window_size = 50
    if len(df_concat) >= window_size:
        df_concat['r_smooth'] = df_concat['r'].rolling(window=window_size).mean()
        
    # 4. Plot
    plt.figure(figsize=(10, 5))
    
    # Use 'l' (episode length) cumsum to estimate total timesteps
    # This aligns the x-axis for multiple environments
    x_axis = df_concat['l'].cumsum()
    
    plt.plot(x_axis, df_concat['r'], alpha=0.3, color='gray', label='Raw Reward')
    
    if 'r_smooth' in df_concat:
        plt.plot(x_axis, df_concat['r_smooth'], color='blue', linewidth=2, label=f'Smoothed ({window_size})')
    
    plt.xlabel("Total Timesteps")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_doraemon_dynamics(doraemon_callback):
    """
    Plots the internal dynamics of Real DORAEMON:
    1. Entropy (How much randomization?)
    2. Lambda (How strict is the constraint?)
    3. Success Rate (Are we meeting the target?)
    """
    history = doraemon_callback.history
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Entropy (The Objective)
    ax1.plot(history['entropy'], color='green', linewidth=2)
    ax1.set_ylabel('Entropy (Sum Log Std)')
    ax1.set_title('Objective: Maximize Entropy')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Lambda (The Lagrangian Multiplier)
    ax2.plot(history['lambda'], color='red', linewidth=2)
    ax2.set_ylabel('Lambda (Penalty)')
    ax2.set_title('Lagrangian Multiplier (Safety Temperature)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Success Rate (The Constraint)
    ax3.plot(history['success'], color='blue', linewidth=2, label='Current Success')
    ax3.axhline(y=0.8, color='black', linestyle='--', label='Target (0.8)')
    ax3.set_ylabel('Success Rate')
    ax3.set_xlabel('Updates')
    ax3.set_title('Constraint: Success >= Target')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()