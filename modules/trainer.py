# modules/trainer.py
# Main training script to setup environment, model, and training loop.
# Uses UDR and DORAEMON as needed.

import os
import sys
import json
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env.custom_hopper import *

# Import your NEW wrapper and callback
from modules.env import GaussianHopperWrapper
from modules.callbacks import DoraemonCallback


def make_wrapped_env(env_id, use_doraemon):
    """Factory for the environment."""
    def _init():
        # Worker-safe import fix
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.append(project_root)
        import env.custom_hopper # Register env in worker
        
        env = gym.make(env_id)
        if use_doraemon:
            env = GaussianHopperWrapper(env, initial_mean=1.0, initial_std=0.01)
        return env
    return _init

def train_agent(config, log_dir="./logs/", resume_step=None):
    """
    Main training loop with Resume capability.
    :param resume_step: If integer (e.g. 50000), loads checkpoint from that step.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    device = 'cpu'
        
    n_envs = config.get('n_envs', 1) if config['vectorize'] else 1
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    
    # 2. Create Environment
    # Note: We create a fresh env first, then load stats into it if resuming
    env = make_vec_env(
        make_wrapped_env(config['env_id'], use_doraemon=True),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
        monitor_dir=log_dir 
    )

    # 3. Initialize Model & Callback Variables
    model = None
    initial_lambda = 1.0
    initial_history = None
    
    # --- BRANCH: RESUME vs NEW ---
    if resume_step is not None:
        print(f"--- RESUMING TRAINING FROM STEP {resume_step} ---")
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        
        # A. Load Normalization Stats
        if config['normalize']:
            path = f"{ckpt_dir}/vecnormalize_{resume_step}.pkl"
            if os.path.exists(path):
                env = VecNormalize.load(path, env)
                env.training = True # Ensure we keep updating stats
                print("Loaded VecNormalize stats.")
            else:
                print("Warning: No VecNormalize checkpoint found (starting fresh stats).")
        
        # B. Load Model & Replay Buffer
        model = SAC.load(f"{ckpt_dir}/model_{resume_step}", env=env, device=device)
        model.load_replay_buffer(f"{ckpt_dir}/replay_buffer_{resume_step}")
        print("Loaded Model and Replay Buffer.")
        
        # C. Load DORAEMON State
        with open(f"{ckpt_dir}/doraemon_state_{resume_step}.json", "r") as f:
            state = json.load(f)
            
        # Apply loaded physics to the env immediately
        env.env_method('set_distribution', state['mean'], state['std'])
        initial_lambda = state['lambda']
        initial_history = state.get('history', None)

        print(f"Restored DORAEMON: Lambda={initial_lambda:.2f}, Mean={state['mean']}, Std={state['std']}")
        
    else:
        # --- NEW TRAINING ---
        print("--- STARTING NEW TRAINING ---")
        if config['normalize']:
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., training=True)
            
        model = SAC(
            'MlpPolicy', 
            env, 
            seed=config['seed'], 
            verbose=1, 
            learning_rate=config["lr"],
            device=device,
            tensorboard_log="./tensorboard_logs/"
        )

    # 4. Setup Callback (With Checkpointing Enabled)
    doraemon_cb = DoraemonCallback(
        training_env=env,
        target_success=0.8,
        buffer_size=10, 
        lr_param=0.05, 
        lr_lambda=0.1,
        # Checkpoint settings
        save_freq=50000,   # Save every 50k steps
        save_path=log_dir,
        initial_lambda=initial_lambda,
        initial_history=initial_history
    )

    # 5. Run Training
    try:
        # If resuming, reset_num_timesteps=False prevents TB from overwriting old logs
        reset_timesteps = (resume_step is None)
        total_steps = config['timesteps']
        
        model.learn(total_timesteps=total_steps, callback=[doraemon_cb], reset_num_timesteps=reset_timesteps)
        
        # Final Save
        model.save(f"{log_dir}/final_model")
        if config['normalize']:
            env.save(f"{log_dir}/final_vecnormalize.pkl")
            
    except KeyboardInterrupt:
        print("Interrupted! Saving emergency checkpoint...")
        doraemon_cb.save_checkpoint()

    return model, env, doraemon_cb