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
from modules.env import GaussianHopperWrapper, BetaHopperWrapper, UDRHopperWrapper
from modules.callbacks import BetaDoraemonCallback


def make_wrapped_env(env_id, mode='source'):
    """Factory for the environment."""
    def _init():
        # Worker-safe import fix
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.append(project_root)
        import env.custom_hopper # Register env in worker
        
        env = gym.make(env_id)
        if mode == 'doraemon-beta':
            env = BetaHopperWrapper(env, initial_alpha=7.0, initial_beta=7.0)
        elif mode == 'doraemon-gaussian':
            env = GaussianHopperWrapper(env, initial_mean=1.0, initial_std=0.01)
        else:
            env = UDRHopperWrapper(env, udr_range=(0.5, 2.0))
        return env
    return _init

def train_agent(config, log_dir="./logs/", model_name="final_model", resume_step=None):
    """
    Main training loop with Resume capability.
    :param resume_step: If integer (e.g. 50000), loads checkpoint from that step.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    device = 'cpu'
        
    n_envs = config.get('n_envs', 1) if config['vectorize'] else 1
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    # DETERMINE MODE
    # If config has "mode", use it. Otherwise check "use_doraemon" for backward compatibility
    mode = config.get('mode', 'source')
    if config.get('use_doraemon', False):
        mode = 'doraemon'

    print(f"Training with mode: {mode}")
    
    # 2. Create Environment
    # Note: We create a fresh env first, then load stats into it if resuming
    env = make_vec_env(
        make_wrapped_env(config['env_id'], mode=mode),
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
        model = SAC.load(f"{ckpt_dir}/model_{resume_step}", env=env, device=device)
        print("Loaded Model.")
        
        # --- MODIFICA QUI ---
        buffer_path = f"{ckpt_dir}/replay_buffer_{resume_step}.pkl"
        if os.path.exists(buffer_path):
            model.load_replay_buffer(buffer_path)
            print("Loaded Replay Buffer successfully.")
        else:
            print(f"WARNING: Replay Buffer {buffer_path} not found!")
            print("Resuming with EMPTY buffer. Performance might drop initially while refilling.")
        # --------------------
        
        # C. Load DORAEMON State
        with open(f"{ckpt_dir}/doraemon_state_{resume_step}.json", "r") as f:
            state = json.load(f)
            
        # Apply loaded physics to the env immediately
        env.env_method('set_beta_distribution', state['alpha'], state['beta'])
        initial_lambda = state['lambda']
        initial_history = state.get('history', None)

        print(f"Restored DORAEMON: Lambda={initial_lambda:.2f}, Alpha={state['alpha']}, Beta={state['beta']}")
        
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
            tensorboard_log="./tensorboard_logs/",
            gradient_steps=config.get('gradient_steps', 1),
            train_freq=config.get('train_freq', 1),
        )


    # SETUP CALLBACKS CONDITIONALLY
    callbacks_list = []
    
    if mode == 'doraemon':
        # Only use DORAEMON callback for the adaptive agent
        doraemon_cb = BetaDoraemonCallback(
            training_env=env,
            target_success=0.65,
            buffer_size=100, 
            lr_param=0.01, 
            lr_lambda=0.01,
            # Checkpoint settings
            save_freq=50000,   # Save every 50k steps
            save_path=log_dir,
            initial_lambda=initial_lambda,
            initial_history=initial_history
        )
        callbacks_list.append(doraemon_cb)
    else:
        # For UDR or Source, we don't need a specific callback
        print(f"Skipping DoraemonCallback for mode: {mode}")
    

    # 5. Run Training
    try:
        # If resuming, reset_num_timesteps=False prevents TB from overwriting old logs
        reset_timesteps = (resume_step is None)
        total_steps = config['timesteps']
        
        model.learn(total_timesteps=total_steps, callback=callbacks_list, reset_num_timesteps=reset_timesteps)
        
        # Final Save
        model.save(f"{log_dir}/{model_name}")
        if config['normalize']:
            env.save(f"{log_dir}/{model_name}_vecnormalize.pkl")
            
    except KeyboardInterrupt:
        print("Interrupted! Saving emergency checkpoint...")
        doraemon_cb.save_checkpoint()

    if mode.startswith('doraemon'):
        return model, env, doraemon_cb
    else:
        return model, env, None