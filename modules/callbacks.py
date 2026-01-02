# modules/callbacks.py
# Callback implementing DORAEMON-Lite to adapt mass randomization based on success rate.
# This handles the "Auto-Tuning" of the environment difficulty.

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

class DoraemonLiteCallback(BaseCallback):
    """
    DORAEMON-Lite: Adapts entropy (mass range width) based on success rate.
    """
    # Note: We removed 'udr_wrapper' from args as discussed
    def __init__(self, eval_env, training_env, target_success=0.8, lr_entropy=0.05, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        
        # === FIX: Rename this variable to avoid conflict with BaseCallback ===
        self.dr_training_env = training_env 
        self.wrapper = training_env 
        
        self.target = target_success
        self.lr = lr_entropy
        self.check_freq = check_freq
        
        self.current_range_width = 0.0 
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # 1. Sync Normalization Stats
            # Use the renamed variable 'self.dr_training_env'
            if isinstance(self.dr_training_env, VecNormalize):
                sync_envs_normalization(self.dr_training_env, self.eval_env)

            # 2. Evaluate
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)
            
            # Heuristic: Reward > 600 is good enough to start expanding
            is_success = 1.0 if mean_reward > 600 else 0.0
            
            # 3. Update Entropy (Gradient Step)
            error = is_success - self.target
            self.current_range_width += self.lr * error
            self.current_range_width = np.clip(self.current_range_width, 0.0, 1.0) 
            
            # 4. Apply to Wrapper
            new_min = 1.0 - (self.current_range_width / 2)
            new_max = 1.0 + (self.current_range_width / 2)
            
            if hasattr(self.wrapper, 'set_attr'):
                self.wrapper.set_attr('mass_range_scale', [new_min, new_max])
                if self.verbose > 0:
                    print(f"[DORAEMON] Updated envs to range [{new_min:.2f}, {new_max:.2f}] (Reward: {mean_reward:.0f})")
            else:
                self.wrapper.mass_range_scale = [new_min, new_max]
            
            self.history.append((self.num_timesteps, self.current_range_width, mean_reward))
                
        return True