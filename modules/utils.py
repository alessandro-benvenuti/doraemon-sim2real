import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

def plot_learning_curve(log_dir, title="Learning Curve"):
    """
    Reads the monitor.csv file produced by SB3 and plots smoothed rewards.
    """
    try:
        # SB3 Monitor logs are CSVs with 2 lines of metadata header
        df = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
    except FileNotFoundError:
        print(f"No log file found at {log_dir}/monitor.csv")
        return

    # Calculate rolling average for smoothness
    window_size = 50
    df['r_smooth'] = df['r'].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df['l'].cumsum(), df['r'], alpha=0.3, color='gray', label='Raw Reward')
    plt.plot(df['l'].cumsum(), df['r_smooth'], color='blue', linewidth=2, label=f'Smoothed ({window_size})')
    
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_doraemon_dynamics(doraemon_callback):
    """
    Plots the history of Entropy expansion (Mass Range Width) vs Performance.
    """
    if not doraemon_callback or not hasattr(doraemon_callback, 'history'):
        print("No Doraemon history found.")
        return

    # History is a list of tuples: (timesteps, width, reward)
    data = np.array(doraemon_callback.history)
    
    if len(data) == 0:
        print("Doraemon history is empty.")
        return

    timesteps = data[:, 0]
    widths = data[:, 1]
    rewards = data[:, 2]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot 1: Entropy (Width)
    color = 'tab:red'
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Mass Range Width (Entropy)', color=color)
    ax1.plot(timesteps, widths, color=color, linewidth=2, label='Entropy (Width)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance (Reward)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Eval Reward', color=color)
    ax2.plot(timesteps, rewards, color=color, linestyle='--', alpha=0.6, label='Eval Reward')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("DORAEMON: Entropy Expansion vs Performance")
    fig.tight_layout()
    plt.show()

def evaluate_sim2real(model, source_env, target_env, n_episodes=20):
    """
    Evaluates the model on Source (Sim) and Target (Real/Proxy)
    """
    print(f"--- Evaluating over {n_episodes} episodes ---")
    
    # 1. Source Performance
    mean_source, std_source = evaluate_policy(model, source_env, n_eval_episodes=n_episodes)
    print(f"Source Env Reward: {mean_source:.2f} +/- {std_source:.2f}")

    # 2. Target Performance
    mean_target, std_target = evaluate_policy(model, target_env, n_eval_episodes=n_episodes)
    print(f"Target Env Reward: {mean_target:.2f} +/- {std_target:.2f}")
    
    gap = mean_source - mean_target
    print(f"Sim2Real Gap: {gap:.2f}")
    
    return mean_source, mean_target