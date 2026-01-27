# modules/env.py
# Wrapper of env to randomize parameters in the environment
# This handles the physics randomization logic.

import gymnasium as gym
import numpy as np

class UDRHopperWrapper(gym.Wrapper):
    """
    Uniform Domain Randomization for Hopper.
    Randomizes: 3 Masses, Global Friction, and Global Damping.
    """
    def __init__(self, env, udr_range=(0.5, 2.0)):
        super().__init__(env)
        self.udr_range = list(udr_range)
        
        # Store original values
        self.original_masses = np.copy(env.unwrapped.model.body_mass)
        self.original_friction = np.copy(env.unwrapped.model.geom_friction)
        self.original_damping = np.copy(env.unwrapped.model.dof_damping)

        # Indices (matching BetaHopperWrapper)
        self.mass_indices = [2, 3, 4] # Thigh, Leg, Foot
        self.geom_indices = [0, 1, 2, 3, 4] # Floor + Robot parts
        self.dof_indices = [3, 4, 5] # Thigh, Leg, Foot joints
        self.n_params = 5 

    def reset(self, **kwargs):
        low, high = self.udr_range
        # Sample 5 multipliers uniformly
        scales = np.random.uniform(low, high, size=self.n_params)
        
        # 1. Apply Masses
        new_masses = np.copy(self.original_masses)
        new_masses[self.mass_indices] *= scales[0:3]
        self.unwrapped.model.body_mass[:] = new_masses

        # 2. Apply Global Friction
        new_friction = np.copy(self.original_friction)
        new_friction[self.geom_indices, 0] *= scales[3] 
        self.unwrapped.model.geom_friction[:] = new_friction
        
        # 3. Apply Global Damping
        new_damping = np.copy(self.original_damping)
        new_damping[self.dof_indices] *= scales[4]
        self.unwrapped.model.dof_damping[:] = new_damping

        return self.env.reset(**kwargs)


class GaussianHopperWrapper(gym.Wrapper):
    """
    Randomizes Thigh(2), Leg(3), Foot(4) masses using a Gaussian distribution.
    DORAEMON learns the Mean and Std of this distribution.
    """
    def __init__(self, env, initial_mean=1.0, initial_std=0.001):
        super().__init__(env)
        
        # 1. Store original physics parameters
        self.original_masses = np.copy(env.unwrapped.model.body_mass)
        self.original_friction = np.copy(env.unwrapped.model.geom_friction)
        self.original_damping = np.copy(env.unwrapped.model.dof_damping)

        # Indices definition
        self.mass_indices = [2, 3, 4] # Thigh, Leg, Foot
        self.geom_indices = [0, 1, 2, 3, 4] # Floor + Robot parts
        self.dof_indices = [3, 4, 5] # Thigh, Leg, Foot joints

        # --- PARAMETER MAPPING ---
        # Param 0,1,2: Masses (Thigh, Leg, Foot)
        # Param 3: Global Friction Scale (Multiplier for all geoms)
        # Param 4: Global Damping Scale (Multiplier for all joints)
        self.n_params = 5 
        
        self.mean = np.full(self.n_params, initial_mean, dtype=np.float32)
        self.std = np.full(self.n_params, initial_std, dtype=np.float32)

    def reset(self, **kwargs):
        # Sample scales from distribution
        scales = np.random.normal(self.mean, self.std)
        # Clip to avoid unstable physics (negative mass/friction)
        scales = np.clip(scales, 0.2, 5.0)        
        self.last_scales = scales 
        
        # --- APPLY PHYSICS ---
        
        # A. Update Masses (Individual control)
        new_masses = np.copy(self.original_masses)
        new_masses[self.mass_indices] *= scales[0:3]
        self.unwrapped.model.body_mass[:] = new_masses

        # B. Update Friction (Global Scale)
        # friction array is (n_geoms, 3) -> sliding, torsional, rolling. We scale sliding (idx 0).
        new_friction = np.copy(self.original_friction)
        new_friction[self.geom_indices, 0] *= scales[3] 
        self.unwrapped.model.geom_friction[:] = new_friction
        
        # C. Update Damping (Global Scale)
        new_damping = np.copy(self.original_damping)
        new_damping[self.dof_indices] *= scales[4]
        self.unwrapped.model.dof_damping[:] = new_damping

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
    

class BetaHopperWrapper(gym.Wrapper):
    def __init__(self, env, initial_alpha=2.0, initial_beta=2.0):
        super().__init__(env)
        self.original_masses = np.copy(env.unwrapped.model.body_mass)
        self.original_friction = np.copy(env.unwrapped.model.geom_friction)
        self.original_damping = np.copy(env.unwrapped.model.dof_damping)

        self.mass_indices = [2, 3, 4]
        self.geom_indices = [0, 1, 2, 3, 4]
        self.dof_indices = [3, 4, 5]
        
        # Fixed physical range (Support of the Beta)
        self.phys_min = 0.5
        self.phys_max = 1.5
        
        self.n_params = 5
        self.alpha = np.full(self.n_params, initial_alpha, dtype=np.float32)
        self.beta = np.full(self.n_params, initial_beta, dtype=np.float32)

    def reset(self, **kwargs):
        # Sample from Beta in the unit interval [0, 1]
        unit_scales = np.random.beta(self.alpha, self.beta)
        self.last_unit_scales = unit_scales # Save the [0, 1] value for the Callback
        
        # Linear transformation to the physical range [0.5, 2.0]
        scales = self.phys_min + (self.phys_max - self.phys_min) * unit_scales
        
        # Apply parameters (same logic as before)
        new_masses = np.copy(self.original_masses)
        new_masses[self.mass_indices] *= scales[0:3]
        self.unwrapped.model.body_mass[:] = new_masses

        new_friction = np.copy(self.original_friction)
        new_friction[self.geom_indices, 0] *= scales[3] 
        self.unwrapped.model.geom_friction[:] = new_friction
        
        new_damping = np.copy(self.original_damping)
        new_damping[self.dof_indices] *= scales[4]
        self.unwrapped.model.dof_damping[:] = new_damping

        return self.env.reset(**kwargs)

    def get_last_scales(self):
        """
        Returns the last sampled scales in the unit interval [0, 1].
        """       
        return self.last_unit_scales

    def set_beta_distribution(self, alpha, beta):
        """
        Called by the DORAEMON callback to update the distribution parameters
        after a gradient step.
        """
        self.alpha, self.beta = alpha, beta

    def get_distribution_params(self):
        """
        Returns current parameters, useful for logging.
        """
        return self.alpha, self.beta
    

