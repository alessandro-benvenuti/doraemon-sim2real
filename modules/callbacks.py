# modules/callbacks.py
# Callback implementing DORAEMON-Lite to adapt mass randomization based on success rate.
# This handles the "Auto-Tuning" of the environment difficulty.

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

class DoraemonCallback(BaseCallback):
    """
    DORAEMON: Optimizes the distribution of environment parameters (masses)
    to Maximize Entropy subject to a Success Rate constraint.
    
    Objective: max J = Entropy + lambda * (SuccessRate - Target)
    """
    def __init__(self, training_env, target_success=0.8, buffer_size=50, lr_param=1e-3, lr_lambda=1e-2, verbose=1):
        super().__init__(verbose)
        self.doraemon_env = training_env
        self.target_success = target_success
        self.buffer_size = buffer_size # How many episodes to collect before an update
        
        # Optimization Hyperparameters
        self.lr_param = lr_param
        self.lr_lambda = lr_lambda
        
        # Initialize Lagrangian Multiplier (Lambda)
        # Higher lambda = more focus on Success (Safety), Lower lambda = more focus on Entropy (Exploration)
        self.labda = 1.0 
        
        # Buffers to store episode data
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
                is_success = 1.0 if reward > 600 else 0.0
                
                # 2. Get the Parameters that generated this outcome
                # We query the specific env instance for the parameters used in the last episode
                last_scales = self.doraemon_env.env_method('get_last_scales', indices=[i])[0]
                
                # 3. Store in Buffer
                self.episode_params.append(last_scales)
                self.episode_outcomes.append(is_success)
                
                # 4. Update Distribution if Buffer is Full
                if len(self.episode_params) >= self.buffer_size:
                    self.update_distribution()
                    
        return True

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
        new_std = np.clip(new_std, 0.001, 1.0)
        
        self.doraemon_env.env_method('set_distribution', new_mean, new_std)
        
        # --- LOGGING ---
        if self.verbose > 0:
            print(f"[DORAEMON] Step {self.num_timesteps}: Success={avg_success:.2f} | Lambda={self.labda:.2f} | Mean={new_mean} | Std={new_std}")
            
        self.history['entropy'].append(np.sum(np.log(new_std)))
        self.history['success'].append(avg_success)
        self.history['lambda'].append(self.labda)
        
        # Clear buffers
        self.episode_params = []
        self.episode_outcomes = []
