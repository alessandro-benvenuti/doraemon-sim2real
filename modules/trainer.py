# modules/trainer.py
# Main training script to setup environment, model, and training loop.
# Uses UDR and DORAEMON as needed.

import os
import sys
import json
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env.custom_hopper import *

# Import your NEW wrapper and callback
from modules.env import (
    GaussianHopperWrapper,
    GaussianCartPoleWrapper,
    GaussianHalfCheetahWrapper,
    UDRHopperWrapper,
    UDRCartPoleWrapper,
    UDRHalfCheetahWrapper,
)
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
        import env.custom_carpole # Register env in worker
        import env.custom_halfcheetah # Register env in worker

        env = gym.make(env_id)
        
        if use_doraemon:
            # DORAEMON: Gaussian distribution for adaptation
            if 'Hopper' in env_id or 'hopper' in env_id:
                env = GaussianHopperWrapper(env, initial_mean=1.0, initial_std=0.01)
            elif 'CartPole' in env_id or 'cartpole' in env_id.lower():
                env = GaussianCartPoleWrapper(env, initial_mean=1.0, initial_std=0.01)
            elif 'HalfCheetah' in env_id or 'halfcheetah' in env_id.lower():
                env = GaussianHalfCheetahWrapper(env, initial_mean=1.0, initial_std=0.01)
        else:
            # UDR: Uniform Domain Randomization
            if 'Hopper' in env_id or 'hopper' in env_id:
                env= UDRHopperWrapper(env)  # Hopper UDR can be added later if needed
            elif 'CartPole' in env_id or 'cartpole' in env_id.lower():
                env = UDRCartPoleWrapper(env)
            elif 'HalfCheetah' in env_id or 'halfcheetah' in env_id.lower():
                env = UDRHalfCheetahWrapper(env)
        
        return env
    return _init

def train_agent(config, log_dir="./logs/", resume_step=None):
    """
    Main training loop with Resume capability.
    :param resume_step: If integer (e.g. 50000), loads checkpoint from that step.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    if config.get('device', 'cpu') == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
        
    n_envs = config.get('n_envs', 1) if config['vectorize'] else 1
    # On Windows, SubprocVecEnv can have issues. Use DummyVecEnv for CartPole
    if 'CartPole' in config['env_id']:
        vec_env_cls = DummyVecEnv  # Force DummyVecEnv for CartPole on Windows
    else:
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    
    # 2. Create Environment
    # Note: We create a fresh env first, then load stats into it if resuming
    env = make_vec_env(
        make_wrapped_env(config['env_id'], use_doraemon=config['use_doraemon']),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
        monitor_dir=log_dir 
    )

    # 3. Initialize Model & Callback Variables
    model = None
    initial_lambda = 5.0
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
        if config['algorithm'] == 'SAC':
            model = SAC.load(f"{ckpt_dir}/model_{resume_step}", env=env, device=device)
            model.load_replay_buffer(f"{ckpt_dir}/replay_buffer_{resume_step}")
            print("Loaded Model and Replay Buffer.")
        
        elif config['algorithm'] == 'PPO':
            model = PPO.load(f"{ckpt_dir}/model_{resume_step}", env=env, device=device)
            
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
        
        model
        if config['algorithm'] == 'SAC':
            model = SAC(
                'MlpPolicy', 
                env, 
                seed=config['seed'], 
                verbose=1, 
                learning_rate=config["lr"],
                device=device,
                tensorboard_log="./tensorboard_logs/"
            )
        elif config['algorithm'] == 'PPO':
            model = PPO(
                'MlpPolicy', 
                env, 
                seed=config['seed'], 
                verbose=1, 
                learning_rate=config["lr"],
                device=device,
                tensorboard_log="./tensorboard_logs/"
            )

        # 4. Setup Callback (Only if DORAEMON is enabled)
        doraemon_cb = None
        callbacks_list = []
            
        if config['use_doraemon']:
            doraemon_cb = DoraemonCallback(
            training_env=env,
            target_success= config.get('target_success', 0.7),
            buffer_size= config.get('buffer_size', 20),
            lr_param= config.get('lr_param', 0.01), 
            lr_lambda= config.get('lr_lambda', 0.5),
            # Checkpoint settings
            save_freq=50000,   # Save every 50k steps
            save_path=log_dir,
            initial_lambda=initial_lambda,
            initial_history=initial_history
            )
            callbacks_list = [doraemon_cb]
        else:
            print("--- UDR Mode: Training without DORAEMON ---")

            # 5. Run Training
        try:
            # If resuming, reset_num_timesteps=False prevents TB from overwriting old logs
            reset_timesteps = (resume_step is None)
            total_steps = config['timesteps']
                
            model.learn(total_timesteps=total_steps, callback=callbacks_list, reset_num_timesteps=reset_timesteps)
                
                # Final Save
            model.save(f"{log_dir}/final_model")
            if config['normalize']:
                env.save(f"{log_dir}/final_vecnormalize.pkl")
                
        except KeyboardInterrupt:
            print("Interrupted! Saving emergency checkpoint...")
            if doraemon_cb is not None:
                doraemon_cb.save_checkpoint()

    return model, env, doraemon_cb