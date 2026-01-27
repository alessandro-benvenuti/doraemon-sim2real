"""Implementation of the CartPole environment supporting
domain randomization optimization."""
import csv
import pdb
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.spaces import Box

# nota : CartPoleEnv ha già implementato reset e step
# quindi qui estendiamo solo per aggiungere le funzionalità di domain randomization
# i parametri fisici di CartPole sono:
# - masscart (massa del carrello)
# - masspole (massa dell'asta)
# - length (metà della lunghezza dell'asta)
# - force_mag (forza applicata al carrello)
# - gravity (accelerazione di gravità)
# - tau (intervallo di integrazione del tempo)
# non sono previsti parametri per il target environment, quindi li scelgo arbitrariamente 
# modifico massa del car e lunghezza del pole
class CustomCartPole(CartPoleEnv, utils.EzPickle):
    """
    Custom CartPole environment with domain randomization support.
    
    Extends the standard CartPole-v1 to allow parameter manipulation
    for domain randomization (masses, length, force magnitude, etc.)
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            render_mode,
            domain,
            **kwargs,
        )
        
        # Initialize parent CartPole
        super().__init__(render_mode=render_mode)
        
        # Store original parameters for reset
        self.original_masscart = self.masscart
        self.original_masspole = self.masspole
        self.original_length = self.length
        self.original_force_mag = self.force_mag
        self.original_gravity = self.gravity
        self.original_tau = self.tau
        
        # Domain-specific modifications
        if domain == 'source':
            # Source environment has slightly different cart mass (e.g., +0.5kg)
            self.masscart += 0.5
            self.total_mass = self.masspole + self.masscart
            self.polemass_length = self.masspole * self.length
            
        elif domain == 'target':
            # Target environment has different parameters
            self.masspole *= 1.2
            self.length *= 0.8
            self.total_mass = self.masspole + self.masscart
            self.polemass_length = self.masspole * self.length

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the environment."""
        observation, info = super().reset(seed=seed, options=options)
        
        # Optional: Sample random parameters on reset
        # Uncomment if you want automatic randomization on reset
        # self.set_random_parameters()
        
        return observation, info

    def set_random_parameters(self):
        """Set random physics parameters.
        This can be called by wrapper classes or explicitly.
        """
        self.set_parameters(*self.sample_parameters())

    def sample_parameters(self):
        """Sample physics parameters according to a domain randomization distribution.
        
        Returns:
            Tuple of (masscart_scale, masspole_scale, length_scale, force_scale)
        """
        # Sample uniform random scales between 0.5 and 2.0
        masscart_scale = np.random.uniform(0.5, 2.0)
        masspole_scale = np.random.uniform(0.5, 2.0)
        length_scale = np.random.uniform(0.7, 1.5)
        force_scale = np.random.uniform(0.8, 1.2)
        
        return masscart_scale, masspole_scale, length_scale, force_scale

    def get_parameters(self):
        """Get current physics parameters as scales relative to original values.
        
        Returns:
            numpy array of [masscart, masspole, length, force_mag]
        """
        params = np.array([
            self.masscart,
            self.masspole,
            self.length,
            self.force_mag
        ])
        return params
    
    def get_parameter_scales(self):
        """Get current physics parameters as scales relative to original values.
        
        Returns:
            numpy array of scale factors
        """
        scales = np.array([
            self.masscart / self.original_masscart,
            self.masspole / self.original_masspole,
            self.length / self.original_length,
            self.force_mag / self.original_force_mag
        ])
        return scales

    def set_parameters(self, masscart_scale, masspole_scale, length_scale, force_scale):
        """Set physics parameters using scale factors.
        
        Args:
            masscart_scale: Scale factor for cart mass
            masspole_scale: Scale factor for pole mass
            length_scale: Scale factor for pole length
            force_scale: Scale factor for force magnitude
        """
        self.masscart = self.original_masscart * masscart_scale
        self.masspole = self.original_masspole * masspole_scale
        self.length = self.original_length * length_scale
        self.force_mag = self.original_force_mag * force_scale
        
        # Update derived parameters
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
    
    def set_parameters_direct(self, masscart, masspole, length, force_mag):
        """Set physics parameters directly (not as scales).
        
        Args:
            masscart: Cart mass in kg
            masspole: Pole mass in kg
            length: Half-pole length in meters
            force_mag: Force magnitude in Newtons
        """
        self.masscart = masscart
        self.masspole = masspole
        self.length = length
        self.force_mag = force_mag
        
        # Update derived parameters
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
    
    def reset_to_original_parameters(self):
        """Reset all physics parameters to their original values."""
        self.masscart = self.original_masscart
        self.masspole = self.original_masspole
        self.length = self.original_length
        self.force_mag = self.original_force_mag
        self.gravity = self.original_gravity
        self.tau = self.original_tau
        
        # Update derived parameters
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    # DORAEMON Methods (for compatibility with wrappers and callbacks)
    def get_last_scales(self):
        """Get the last parameter scales used (for DORAEMON)."""
        if not hasattr(self, 'last_scales'):
            self.last_scales = np.ones(4)
        return self.last_scales

    def set_distribution(self, mean, std):
        """Set Gaussian distribution parameters (for DORAEMON wrapper)."""
        self.doraemon_mean = np.array(mean, dtype=np.float32)
        self.doraemon_std = np.array(std, dtype=np.float32)

    def get_distribution_params(self):
        """Get current distribution parameters (for DORAEMON callback)."""
        if not hasattr(self, 'doraemon_mean'):
            self.doraemon_mean = np.ones(4, dtype=np.float32)
        if not hasattr(self, 'doraemon_std'):
            self.doraemon_std = np.full(4, 0.001, dtype=np.float32)
        return self.doraemon_mean, self.doraemon_std


"""
    Registered environments
"""

gym.register(
    id="CustomCartPole-v0",
    entry_point="%s:CustomCartPole" % __name__,
    max_episode_steps=500,
)

gym.register(
    id="CustomCartPole-source-v0",
    entry_point="%s:CustomCartPole" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.register(
    id="CustomCartPole-target-v0",
    entry_point="%s:CustomCartPole" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)