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

        # --- PARAMETER MAPPING ---
        # Param 0,1,2: Masses (Thigh, Leg, Foot)
        self.n_params = 3 
        
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
    def __init__(self, env, mass_range=(0.2, 2.0), friction_range=(0.2, 2.0)):
        super().__init__(env)
        self.mass_range = mass_range
        self.friction_range = friction_range

        # Indici: 2-8 (esclude world e torso)
        self.original_masses = np.copy(env.unwrapped.model.body_mass)
        self.mass_indices = list(range(2, len(self.original_masses)))
        
        # Frizione originale del suolo
        self.original_friction = np.copy(env.unwrapped.model.geom_friction[0][0])

    def reset(self, **kwargs):
        # Campionamento asimmetrico: un valore diverso per ogni indice di massa
        mass_scales = np.random.uniform(self.mass_range[0], self.mass_range[1], size=len(self.mass_indices))
        # Campionamento frizione
        friction_scale = np.random.uniform(self.friction_range[0], self.friction_range[1])

        # Applica Masse
        new_masses = np.copy(self.original_masses)
        new_masses[self.mass_indices] *= mass_scales
        self.unwrapped.model.body_mass[:] = new_masses

        # Applica Frizione (Floor geom index 0)
        self.unwrapped.model.geom_friction[0][0] = self.original_friction * friction_scale

        return self.env.reset(**kwargs)


class GaussianHalfCheetahWrapper(gym.Wrapper):
    def __init__(self, env, initial_mean=1.0, initial_std=0.1):
        super().__init__(env)

        self.original_masses = np.copy(env.unwrapped.model.body_mass)
        self.mass_indices = list(range(2, len(self.original_masses)))
        self.original_friction = np.copy(env.unwrapped.model.geom_friction[0][0])

        # Parametri: 6 masse (arti) + 1 frizione = 7 parametri totali
        # (Se includi il torso sono 8, ma meglio tenerlo fisso per il test)
        self.n_params = len(self.mass_indices) + 1
        
        self.mean = np.full(self.n_params, initial_mean, dtype=np.float32)
        self.std = np.full(self.n_params, initial_std, dtype=np.float32)
        self.last_scales = None

    def reset(self, **kwargs):
        # Campionamento asimmetrico da distribuzione Gaussiana
        scales = np.random.normal(self.mean, self.std)
        # Clip per evitare valori fisicamente instabili
        scales = np.clip(scales, 0.05, 5.0) 
        self.last_scales = scales

        # 1. Update Masse (indici 0 a n-2 di scales)
        new_masses = np.copy(self.original_masses)
        new_masses[self.mass_indices] *= scales[:-1]
        self.unwrapped.model.body_mass[:] = new_masses

        # 2. Update Frizione (ultimo indice di scales)
        self.unwrapped.model.geom_friction[0][0] = self.original_friction * scales[-1]

        return self.env.reset(**kwargs)

    def get_last_scales(self):
        return self.last_scales

    def set_distribution(self, mean, std):
        # Importante: assicurati che la lunghezza di mean/std corrisponda a n_params
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def get_distribution_params(self):
        return self.mean, self.std
