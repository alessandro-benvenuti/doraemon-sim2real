# modules/env.py
# Wrapper of env to randomize parameters in the environment
# This handles the physics randomization logic.

import gymnasium as gym
import numpy as np



#--------------------- HOPPER WRAPPERS ---------------------#

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

#--------------------- CARTPOLE WRAPPERS ---------------------#

class UDRCartPoleWrapper(gym.Wrapper):
    """
    Randomizes CartPole parameters (masscart, masspole, length, force_mag)
    with uniform distribution. Simple UDR without DORAEMON.
    param_range: Dict with keys 'masscart', 'masspole', 'length', 'force_mag'
                 each containing tuple (min_scale, max_scale)
    """
    def __init__(self, env, param_range=None):
        super().__init__(env)
        
        # Store original parameters
        self.original_masscart = env.unwrapped.masscart
        self.original_masspole = env.unwrapped.masspole
        self.original_length = env.unwrapped.length
        self.original_force_mag = env.unwrapped.force_mag
        
    # UDR ranges <- FORSE TROPPO ELEVATE
        if param_range is None:
            param_range = {
                'masscart': (0.5, 2.0),
                'masspole': (0.5, 2.0),
                'length': (0.5, 2.0),
                'force_mag': (0.5, 2.0)
            }
        self.param_range = param_range

    def reset(self, **kwargs):
        # Sample uniform scales for each parameter
        masscart_scale = np.random.uniform(*self.param_range['masscart'])
        masspole_scale = np.random.uniform(*self.param_range['masspole'])
        length_scale = np.random.uniform(*self.param_range['length'])
        force_scale = np.random.uniform(*self.param_range['force_mag'])
        
        # Apply to CartPole
        self.unwrapped.masscart = self.original_masscart * masscart_scale
        self.unwrapped.masspole = self.original_masspole * masspole_scale
        self.unwrapped.length = self.original_length * length_scale
        self.unwrapped.force_mag = self.original_force_mag * force_scale
        
        # Update derived quantities
        self.unwrapped.total_mass = self.unwrapped.masspole + self.unwrapped.masscart
        self.unwrapped.polemass_length = self.unwrapped.masspole * self.unwrapped.length
        
        return self.env.reset(**kwargs)
    
    def set_udr_range(self, new_range):
        """Update UDR parameter ranges."""
        self.param_range = new_range

class GaussianCartPoleWrapper(gym.Wrapper):
    """
    Randomizes CartPole parameters (masscart, masspole, length, force_mag) 
    using a Gaussian distribution. DORAEMON learns the Mean and Std.
    """
    def __init__(self, env, initial_mean=1.0, initial_std=0.001):
        super().__init__(env)
        
        # Store original parameters
        self.original_masscart = env.unwrapped.masscart
        self.original_masspole = env.unwrapped.masspole
        self.original_length = env.unwrapped.length
        self.original_force_mag = env.unwrapped.force_mag
        
        # Initialize distribution parameters (4 dimensions for CartPole)
        n_params = 4
        self.mean = np.full(n_params, initial_mean, dtype=np.float32)
        self.std = np.full(n_params, initial_std, dtype=np.float32)
        
        # Store in unwrapped env so it's always accessible
        self.unwrapped.doraemon_mean = self.mean.copy()
        self.unwrapped.doraemon_std = self.std.copy()

    def reset(self, **kwargs):
        scales = np.random.normal(self.mean, self.std)
        scales = np.clip(scales, 0.1, 10.0)  # Prevent extreme values
        self.last_scales = scales
        
        # Store in unwrapped env so callback can access it
        self.unwrapped.last_scales = scales
        
        # Apply scales to CartPole parameters
        self.unwrapped.masscart = self.original_masscart * scales[0]
        self.unwrapped.masspole = self.original_masspole * scales[1]
        self.unwrapped.length = self.original_length * scales[2]
        self.unwrapped.force_mag = self.original_force_mag * scales[3]
        
        # Update derived quantities
        self.unwrapped.total_mass = self.unwrapped.masspole + self.unwrapped.masscart
        self.unwrapped.polemass_length = self.unwrapped.masspole * self.unwrapped.length
        
        return self.env.reset(**kwargs)

    def get_last_scales(self):
        return self.last_scales

    def set_distribution(self, mean, std):
        """
        Called by DORAEMON callback to update distribution parameters.
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        # Also store in unwrapped env
        self.unwrapped.doraemon_mean = self.mean.copy()
        self.unwrapped.doraemon_std = self.std.copy()
        
    def get_distribution_params(self):
        """
        Returns current parameters.
        """
        return self.mean, self.std
    

#--------------------- HALFCHEETAH WRAPPERS ---------------------#

class UDRHalfCheetahWrapper(gym.Wrapper):
    """
    Uniform Domain Randomization for HalfCheetah masses.
    Randomizes all body masses except the world/root body (index 0).
    mass_range_scale: (min, max) multiplier for the masses.
    """
    def __init__(self, env, mass_range_scale=(0.8, 1.2)):
        super().__init__(env)
        self.mass_range_scale = list(mass_range_scale)

        # Store original masses and choose indices to randomize (exclude index 0)
        self.original_masses = np.copy(env.unwrapped.model.body_mass)
        # gemini consiglia di tenere fisso il torso (indice 1) e di randomizzare da 2 in poi list(range(2, len(self.original_masses)))
        self.mass_indices = list(range(1, len(self.original_masses)))

    def reset(self, **kwargs):
        low, high = self.mass_range_scale
        scales = np.random.uniform(low, high, size=len(self.mass_indices))

        new_masses = np.copy(self.original_masses)
        new_masses[self.mass_indices] *= scales
        self.unwrapped.model.body_mass[:] = new_masses

        return self.env.reset(**kwargs)

    def set_udr_range(self, new_range):
        self.mass_range_scale = list(new_range)


class GaussianHalfCheetahWrapper(gym.Wrapper):
    """
    DORAEMON-style Gaussian randomization for HalfCheetah masses.
    Learns mean/std for multiplicative scales applied to body masses
    (all bodies except root/world).
    """
    def __init__(self, env, initial_mean=1.0, initial_std=0.001):
        super().__init__(env)

        # Store original masses and indices to modify (exclude index 0)
        self.original_masses = np.copy(env.unwrapped.model.body_mass)
        self.mass_indices = list(range(1, len(self.original_masses)))

        n_params = len(self.mass_indices)
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
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def get_distribution_params(self):
        return self.mean, self.std


