# modules/env.py
# Wrapper of env to randomize parameters in the environment
# This handles the physics randomization logic.

import gymnasium as gym
import numpy as np

class UDRHopperWrapper(gym.Wrapper):
    """
    Randomizes Thigh(2), Leg(3), Foot(4) masses.
    mass_range_scale: Tuple (min, max) multiplier for the masses.
    """
    def __init__(self, env, mass_range_scale=(0.8, 1.2)):
        super().__init__(env)
        # Convert to list so it is mutable (needed for DORAEMON)
        self.mass_range_scale = list(mass_range_scale)
        
        self.original_masses = np.copy(env.unwrapped.model.body_mass) # Store original masses
        self.mass_indices = [2, 3, 4] # Thigh, Leg, Foot

    def reset(self, **kwargs):
        low, high = self.mass_range_scale
        
        # Sample uniform distribution
        scales = np.random.uniform(low, high, size=len(self.mass_indices))
        
        new_masses = np.copy(self.original_masses)
        new_masses[self.mass_indices] *= scales
        self.unwrapped.model.body_mass[:] = new_masses

        return self.env.reset(**kwargs)
    
    def set_udr_range(self, new_range):
        """Helper method to set new UDR range from outside."""
        self.mass_range_scale = list(new_range)


class GaussianHopperWrapper(gym.Wrapper):
    """
    Randomizes Thigh(2), Leg(3), Foot(4) masses using a Gaussian distribution.
    DORAEMON learns the Mean and Std of this distribution.
    """
    def __init__(self, env, initial_mean=1.0, initial_std=0.001):
        super().__init__(env)
        
        # Store original physics parameters
        self.original_masses = np.copy(env.unwrapped.model.body_mass)
        self.mass_indices = [2, 3, 4] # Thigh, Leg, Foot indices in MuJoCo
        
        # Initialize distribution parameters for each dimension independently
        # Shape is (3,) because we are randomizing 3 bodies
        n_params = len(self.mass_indices)
        
        # Initialize Mean at 1.0 (original mass) and Std close to 0 (deterministic)
        self.mean = np.full(n_params, initial_mean, dtype=np.float32)
        self.std = np.full(n_params, initial_std, dtype=np.float32)

    def reset(self, **kwargs):
        scales = np.random.normal(self.mean, self.std)
        scales = np.clip(scales, 0.1, 10.0)        
        self.last_scales = scales 
        
        new_masses = np.copy(self.original_masses)
        new_masses[self.mass_indices] *= scales
        self.unwrapped.model.body_mass[:] = new_masses

        return self.env.reset(**kwargs)

    def get_last_scales(self):
        return self.last_scales

    def set_distribution(self, mean, std):
        """
        Called by the DORAEMON callback to update the distribution parameters
        after a gradient step.
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
    def get_distribution_params(self):
        """
        Returns current parameters, useful for logging.
        """
        return self.mean, self.std
    

