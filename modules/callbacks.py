# modules/callbacks.py
# Callback implementing DORAEMON-Lite to adapt mass randomization based on success rate.
# This handles the "Auto-Tuning" of the environment difficulty.

import os
import json
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback

class GaussianDoraemonCallback(BaseCallback):
    """
    DORAEMON: Optimizes the distribution of environment parameters (masses)
    to Maximize Entropy subject to a Success Rate constraint.
    
    Objective: max J = Entropy + lambda * (SuccessRate - Target)
    """
    def __init__(self, training_env, threshold_reward, target_success=0.8, buffer_size=50, 
                lr_param=1e-3, lr_lambda=1e-2, verbose=1,
                # Checkpointing arguments
                save_freq=0, save_path=None, initial_lambda=1.0, initial_history=None, min_std=0.001):
        
        super().__init__(verbose)
        self.doraemon_env = training_env
        self.target_success = target_success
        self.buffer_size = buffer_size
        self.lr_param = lr_param
        self.lr_lambda = lr_lambda
        self.min_std = min_std
        # Initialize Lambda (Allow restoring from checkpoint)
        self.labda = initial_lambda

        # Track if we have finished warming up
        self.warmup_complete = False 
        self.warmup_threshold = target_success  # Wait for x% success before starting DORAEMON
        
        self.threshold_reward=threshold_reward
        # Checkpointing setup
        self.save_freq = save_freq
        self.save_path = save_path

        # RESTORE HISTORY IF RESUMING, ELSE START FRESH
        if initial_history is not None:
            self.history = initial_history
            # If resuming, check if we were already successful enough to skip warmup
            if len(self.history['success']) > 0 and self.history['success'][-1] >= self.warmup_thresholds:
                self.warmup_complete = True
        else:
            self.history = {'entropy': [], 'success': [], 'lambda': []}
        
        # Buffers
        self.episode_params = []
        self.episode_outcomes = []
        self.history = {'entropy': [], 'success': [], 'lambda': []}

    def _on_step(self) -> bool:
        # Check if any environment in the vectorized env is done
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for i, done in enumerate(dones):
            if done:
                # 1. Get the Outcome (Success/Failure)
                # We assume Success if Reward > Threshold (e.g. 600 for Hopper)
                # You might need to adjust this threshold based on your specific task
                reward = infos[i].get('episode', {}).get('r', 0)
                is_success = 1.0 if reward > self.threshold_reward else 0.0
                
                # 2. Get the Parameters that generated this outcome
                # We query the specific env instance for the parameters used in the last episode
                last_scales = self.doraemon_env.env_method('get_last_scales', indices=[i])[0]
                
                # 3. Store in Buffer
                self.episode_params.append(last_scales)
                self.episode_outcomes.append(is_success)
                
                # 4. Update Distribution if Buffer is Full
                if len(self.episode_params) >= self.buffer_size:
                    self._handle_buffer_full()
        
        # 5. Save Checkpoint if needed
        if self.save_freq > 0 and self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            print(f"[DORAEMON] Saving Checkpoint at step {self.num_timesteps}...")
            self.save_checkpoint()
                    
        return True
    

    def _handle_buffer_full(self):
        """
        Decides whether to Update Distribution or Keep Waiting based on success.
        """
        current_success_rate = np.mean(self.episode_outcomes)

        # CASE 1: Still Warming Up
        if not self.warmup_complete:
            if current_success_rate >= self.warmup_threshold:
                # Success! Turn on DORAEMON
                print(f"\n[DORAEMON] Warmup Complete! Success ({current_success_rate:.2f}) >= Threshold. Activating.")
                self.warmup_complete = True
                self.update_distribution()
            else:
                # Failure. Reset buffer and keep training on static environment.
                if self.verbose > 0:
                    print(f"[DORAEMON] Warmup: Current Success {current_success_rate:.2f} < {self.warmup_threshold}. Staying static.")
                
                # Clear buffer so we can collect fresh data
                self.episode_params = []
                self.episode_outcomes = []
        
        # CASE 2: Already Warmed Up
        else:
            self.update_distribution()


    def update_distribution(self):
        """
        Performs the Gradient Step (REINFORCE) to update Mean, Std, and Lambda.
        """
        # 1. Retrieve current distribution parameters from Env
        # We assume all envs share the same distribution, so we just ask the first one
        current_mean_np, current_std_np = self.doraemon_env.env_method('get_distribution_params', indices=[0])[0]
        
        # Convert to PyTorch tensors with gradient tracking
        # We optimize Log-Std to ensure Std remains positive
        mu = torch.tensor(current_mean_np, dtype=torch.float32, requires_grad=True)
        log_sigma = torch.tensor(np.log(current_std_np), dtype=torch.float32, requires_grad=True)
        
        # Convert buffer to tensors
        samples = torch.tensor(np.array(self.episode_params), dtype=torch.float32)
        successes = torch.tensor(np.array(self.episode_outcomes), dtype=torch.float32)
        
        # --- CALCULATE GRADIENTS ---
        
        # A. Entropy of Gaussian: H = 0.5 * sum(1 + log(2*pi) + 2*log_sigma)
        # We want to MAXIMIZE entropy
        entropy = torch.sum(log_sigma) # Constant terms don't affect gradient
        
        # B. Expected Success (Constraint)
        # Since the simulator is non-differentiable, we use the Log-Derivative trick (REINFORCE)
        # J_success = E[Success]
        # grad(J_success) approx mean( (Success - Baseline) * grad(log_prob) )
        
        dist = torch.distributions.Normal(mu, torch.exp(log_sigma))
        log_probs = dist.log_prob(samples).sum(dim=1) # Sum over the 3 mass dimensions
        
        # Baseline subtraction to reduce variance
        baseline = successes.mean()
        reinforce_loss = (successes - baseline) * log_probs
        
        # Total Objective: Maximize H + lambda * (Success)
        # We minimize the Negative Objective
        # Note: 'detach()' on lambda because we update it separately
        loss = - (entropy + self.labda * reinforce_loss.mean())
        
        # --- UPDATE PARAMETERS (mu, sigma) ---
        optimizer = torch.optim.Adam([mu, log_sigma], lr=self.lr_param)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- UPDATE LAMBDA (Dual Ascent) ---
        # If Success < Target, Lambda increases (Penalty increases, forces distribution to shrink)
        # If Success > Target, Lambda decreases (Allows more exploration/entropy)
        avg_success = successes.mean().item()
        error = self.target_success - avg_success
        self.labda += self.lr_lambda * error
        self.labda = max(0.0, self.labda) # Lambda must be non-negative
        
        # --- PUSH UPDATES BACK TO ENV ---
        new_mean = mu.detach().numpy()
        new_std = torch.exp(log_sigma).detach().numpy()
        
        # Safety Clip (prevent extreme physics)
        new_mean = np.clip(new_mean, 0.5, 2.0)
        new_std = np.clip(new_std, self.min_std, 0.4)
        
        self.doraemon_env.env_method('set_distribution', new_mean, new_std)
        
        # --- LOGGING ---
        if self.verbose > 0:
            print(f"[DORAEMON] Step {self.num_timesteps}: Success={avg_success:.2f} | Lambda={self.labda:.2f} | Mean={new_mean} | Std={new_std}")
            
        self.history['entropy'].append(float(np.sum(np.log(new_std))))
        self.history['success'].append(avg_success)
        self.history['lambda'].append(self.labda)
        
        # Clear buffers
        self.episode_params = []
        self.episode_outcomes = []

    def save_checkpoint(self):
        """Saves Model, ReplayBuffer, VecNormalize stats, and DORAEMON state."""
        save_dir = os.path.join(self.save_path, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        step = self.num_timesteps
        
        print(f"Saving checkpoint at step {step}...")

        # 1. Get current physics params
        mean, std = self.doraemon_env.env_method('get_distribution_params', indices=[0])[0]

        sanitized_history = {
            key: [float(x) for x in value_list] 
            for key, value_list in self.history.items()
        }
        
        # 2. Save DORAEMON specific state (JSON)
        state = {
            "timesteps": step,
            "lambda": float(self.labda),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "history": sanitized_history
        }
        with open(f"{save_dir}/doraemon_state_{step}.json", "w") as f:
            json.dump(state, f, indent=4)
            
        # 3. Save SB3 Components
        self.model.save(f"{save_dir}/model_{step}")
        self.model.save_replay_buffer(f"{save_dir}/replay_buffer_{step}")
        
        # 4. Save Normalization Stats (Critical!)
        if hasattr(self.doraemon_env, 'save'):
            self.doraemon_env.save(f"{save_dir}/vecnormalize_{step}.pkl")
