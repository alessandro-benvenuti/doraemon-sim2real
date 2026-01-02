# modules/trainer.py
# Main training script to setup environment, model, and training loop.
# Uses UDR and DORAEMON as needed.

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from .env import UDRHopperWrapper
from .callbacks import DoraemonLiteCallback

def make_wrapped_env(env_id, use_udr, udr_range):
    """
    Internal function that creates a single environment and applies the UDR Wrapper.
    This function is passed to make_vec_env which will execute it N times.
    """
    def _init():
        env = gym.make(env_id)
        if use_udr:
        # Wrap with UDR --> This is a lighter version of DORAEMON
        # In real DORAEMON we would have another distribution instead of the fixed one.
        # We always need a Wrapper (to change physics) and a Callback (to decide how to change them). 
        # In Lite, we recycle the UDR wrapper from lab 4 because it's the easiest one to code.
            env = UDRHopperWrapper(env, mass_range_scale=udr_range)
        return env
    return _init

def train_agent(config, log_dir="./logs/"):
    # 1. Training Env (Vectorized & Normalized)
    n_envs = 8 if config['vectorize'] else 1
    
    env = make_vec_env(
        make_wrapped_env(config['env_id'], config['use_udr'], config['udr_initial_range']),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )
    
    if config['normalize']:
        # Training = True (Update stats as we learn)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Evaluation Env (MUST MATCH TRAINING ENV!)
    # We create a separate env for the callback to test on
    eval_env = make_vec_env(
        make_wrapped_env(config['env_id'], config['use_udr'], config['udr_initial_range']),
        n_envs=1, # Eval is usually on 1 env
        vec_env_cls=DummyVecEnv
    )
    
    if config['normalize']:
        # Training = False (Do not update stats during test, just use existing ones)
        # norm_reward = False (We want to see the REAL reward, not the scaled one)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)

    # Setup Callback Doraemon
    callbacks = []
    if config['use_doraemon']:
        eval_env = make_vec_env(
            make_wrapped_env(config['env_id'], config['use_udr'], config['udr_initial_range']),
            n_envs=1, # Eval only needs 1 env
            vec_env_cls=DummyVecEnv
        )
        if config['normalize']:
            # training=False: Don't update stats during test
            # norm_reward=False: We want to see the REAL unscaled reward
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
            
        doraemon_cb = DoraemonLiteCallback(eval_env, env, target_success=0.8) # We pass 'env' (the vectorized one) to the callback
        callbacks.append(doraemon_cb)

    # model setup
    model_name = f"{config['algo']}_vec_norm_{config['seed']}"
    
    if config['algo'] == 'sac':
        # SAC handles vectorization natively now
        model = SAC('MlpPolicy', env, seed=config['seed'], verbose=1, learning_rate=config["lr"])
    
    print(f"Starting training on {n_envs} parallel environments with Normalization...")
    try:
        model.learn(total_timesteps=config['timesteps'], callback=callbacks)
        
        # Also save normalization statistics!
        # Without this, the loaded model will be dumb because it expects normalized data
        if config['normalize']:
            env.save(f"{log_dir}/{model_name}_vecnormalize.pkl")
            
        model.save(f"{log_dir}/{model_name}")
    except KeyboardInterrupt:
        model.save(f"{log_dir}/{model_name}_interrupted")
        
    return model, callbacks