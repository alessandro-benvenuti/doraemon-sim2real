# modules/utils.py
# Utility functions for plotting learning curves and DORAEMON dynamics.

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def plot_doraemon_dynamics_beta(doraemon_callback):
    """
    Plots DORAEMON dynamics adapting to available data.
    If Alpha/Beta are missing (old checkpoints), it plots only Success, Lambda, and Entropy.
    """
    history = doraemon_callback.history
    
    # 1. Verifica quali dati abbiamo
    has_params = ('alpha_mean' in history) and (len(history['alpha_mean']) > 0)
    
    # 2. Configura il layout (3 o 4 righe)
    rows = 4 if has_params else 3
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3.5 * rows), sharex=True)
    
    # Gestione sicura degli assi
    if rows == 1: axes = [axes] # Caso limite
    
    ax_success = axes[0]
    ax_lambda  = axes[1]
    ax_entropy = axes[2]
    ax_params  = axes[3] if has_params else None

    # --- Plot 1: Success Rate ---
    success = np.array(history.get('success', []))
    if len(success) > 0:
        ax_success.plot(success, color='blue', linewidth=2, label='Current Success')
        # Smoothing opzionale per leggibilità
        if len(success) > 20:
            smooth = pd.Series(success).rolling(20).mean()
            ax_success.plot(smooth, color='cyan', linestyle=':', label='Smoothed')
            
    # Linea del Target
    target = getattr(doraemon_callback, 'target_success', 0.65) # Fallback se manca attributo
    ax_success.axhline(y=target, color='red', linestyle='--', label=f'Target ({target})')
    ax_success.set_ylabel('Success Rate')
    ax_success.set_title('1. Constraint: Success Rate')
    ax_success.legend(loc='lower right')
    ax_success.grid(True, alpha=0.3)

    # --- Plot 2: Lambda ---
    lambdas = history.get('lambda', [])
    ax_lambda.plot(lambdas, color='darkred', linewidth=2)
    ax_lambda.set_ylabel('Lambda')
    ax_lambda.set_title('2. Lagrangian Multiplier (Penalty)')
    ax_lambda.grid(True, alpha=0.3)

    # --- Plot 3: Entropy ---
    entropy = history.get('entropy', [])
    ax_entropy.plot(entropy, color='green', linewidth=2)
    ax_entropy.set_ylabel('Entropy')
    ax_entropy.set_title('3. Objective: Maximize Entropy')
    ax_entropy.grid(True, alpha=0.3)
    
    if not has_params:
        ax_entropy.set_xlabel('Updates') # Se è l'ultimo grafico, metti l'etichetta X qui

    # --- Plot 4: Parameters (Solo se esistono) ---
    if has_params:
        alphas = np.array(history['alpha_mean'])
        betas = np.array(history['beta_mean'])
        
        ax_params.plot(alphas, label='Alpha Mean', color='purple')
        ax_params.plot(betas, label='Beta Mean', color='orange')
        
        # Plot secondario per la media della distribuzione
        dist_mean = alphas / (alphas + betas)
        ax_right = ax_params.twinx()
        ax_right.plot(dist_mean, color='black', linestyle='--', alpha=0.5, label='Dist. Mean')
        ax_right.set_ylabel('Mean (0-1)')
        ax_right.set_ylim(0, 1)
        
        ax_params.set_ylabel('Value')
        ax_params.set_title('4. Distribution Parameters')
        ax_params.set_xlabel('Updates')
        ax_params.legend(loc='upper left')
        ax_params.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def evaluate_sim2real(model, source_env_raw, target_env_raw, log_dir, model_name, n_eval_episodes=20):
    """
    Evaluates the model on Source vs Target and prints Average Reward AND Average Length.
    """
    source_vec = DummyVecEnv([lambda: source_env_raw])
    target_vec = DummyVecEnv([lambda: target_env_raw])

    norm_path = f"{log_dir}/{model_name}_vecnormalize.pkl"
    try:
        source_vec = VecNormalize.load(norm_path, source_vec)
        target_vec = VecNormalize.load(norm_path, target_vec)
        
        source_vec.training = False
        source_vec.norm_reward = False
        target_vec.training = False
        target_vec.norm_reward = False
        
        print(f"Loaded Normalization stats from {norm_path}")
    except FileNotFoundError:
        print("Warning: No normalization stats found. Using RAW obs.")

    def print_stats(name, env):
        rewards, lengths = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=n_eval_episodes, 
            deterministic=True, 
            return_episode_rewards=True
        )
        
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        mean_l, std_l = np.mean(lengths), np.std(lengths)
        
        print(f"\n--- Evaluating on {name} ---")
        print(f"Reward: {mean_r:.2f} +/- {std_r:.2f}")
        print(f"Length: {mean_l:.2f} +/- {std_l:.2f} steps")
        return mean_r

    mean_reward = print_stats("SOURCE Env (Simulation)", source_vec)
    mean_reward_real = print_stats("TARGET Env (Real/Shifted)", target_vec)
    
    return mean_reward, mean_reward_real
# modules/utils.py
# Utility functions for plotting learning curves and DORAEMON dynamics.



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
