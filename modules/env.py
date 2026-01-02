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