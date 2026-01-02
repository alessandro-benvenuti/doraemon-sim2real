# modules/trainer.py
# Main training script to setup environment, model, and training loop.
# Uses UDR and DORAEMON as needed.

import os
import sys
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env.custom_hopper import *

# Import your NEW wrapper and callback
from modules.env import GaussianHopperWrapper
from modules.callbacks import DoraemonCallback


def make_wrapped_env(env_id, use_doraemon, log_dir=None):
    """
    Creates the environment.
    If use_doraemon is True, we wrap it with the Gaussian Wrapper.
    """
    def _init():
        # Used for SubprocVecEnv to create envs in subprocesses
        # -----------------------
        # 1. Add the project root to sys.path inside this worker process
        # This allows the worker to find 'env.custom_hopper'
        current_dir = os.path.dirname(os.path.abspath(__file__)) # .../modules/
        project_root = os.path.dirname(current_dir)              # .../project/
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        # 2. Register the environment locally in the worker
        import env.custom_hopper 
        # -----------------------


        env = gym.make(env_id)
        if use_doraemon:
            # Initialize with standard mass (mean=1.0) and almost zero noise (std=0.01)
            env = GaussianHopperWrapper(env, initial_mean=1.0, initial_std=0.01)

        if log_dir:
            env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

        return env
    return _init

def train_agent(config, log_dir="./logs/"):
    # 1. Setup Environment
    n_envs = 4 if config['vectorize'] else 1

    env = make_vec_env(
        make_wrapped_env(config['env_id'], use_doraemon=True, log_dir=log_dir),
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        monitor_dir=log_dir
    )
    
    # 2. Add Normalization (Optional but recommended for Hopper)
    if config['normalize']:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., training=True)

    # 3. Setup the Real DORAEMON Callback
    # This callback will hold the reference to 'env' and update its Gaussian parameters
    callbacks = []
    doraemon_cb = DoraemonCallback(
        training_env=env,       # The vectorized env wrapper
        target_success=0.8,     # Constraint: Keep success rate >= 80%
        buffer_size=10,         # Update distribution every 10 episodes
        lr_param=0.05,          # Learning rate for Mean/Std
        lr_lambda=0.1          # Learning rate for the Lagrangian multiplier
    )
    callbacks.append(doraemon_cb)

    # 4. Setup Model (SAC)
    model_name = f"{config['algo']}_doraemon_{config['seed']}"
    
    model = SAC(
        'MlpPolicy', 
        env, 
        seed=config['seed'], 
        verbose=1, 
        learning_rate=config["lr"]
    )
    
    print(f"Starting Real DORAEMON training...")
    
    try:
        model.learn(total_timesteps=config['timesteps'], callback=callbacks)
        
        # Save model and normalization stats
        if config['normalize']:
            env.save(f"{log_dir}/{model_name}_vecnormalize.pkl")
        model.save(f"{log_dir}/{model_name}")
        print(f"Model saved at {log_dir}/{model_name}")
        
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        model.save(f"{log_dir}/{model_name}_interrupted")

    return model, env, doraemon_cb