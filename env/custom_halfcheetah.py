"""Implementation of the HalfCheetah environment supporting
domain randomization optimization.
"""
import os
from copy import deepcopy
from typing import Optional, Dict, Union, Tuple

from gymnasium.spaces import Box
import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class CustomHalfCheetah(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 25,
        # fps set from parent via dt; kept here for consistency
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.28, 1.0),
        healthy_angle_range: Tuple[float, float] = (-1.5, 1.5),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None,
        mass_shift: float = 0.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if xml_file == "half_cheetah.xml":
             xml_file = os.path.join(os.path.dirname(__file__), "assets/half_cheetah.xml")


        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        #The observation space consists of the following parts (in order):
        # - qpos (8 elements by default): Position values of the robot’s body parts.
        # - qvel (9 elements): The velocities of these individual body parts (their derivatives).
        #   size 17 (rootz, rooty, rootx, 6 joint angles, bthigh, bshin, bfoot, fthigh, fshiin, ffoot)
        
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation

        )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float64,
        )
        # observation structure 
        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        # Keep a copy of original masses (useful for wrappers or potential manual DR)
        #self.original_masses = np.copy(self.model.body_mass)

        # Simple domain toggle to mirror Hopper/CartPole structure (optional)
        # 1. Identifica chiaramente la massa base dal file XML appena caricato
        base_torso_mass = self.model.body_mass[1]

        if domain == "source":
            # La sorgente è il ghepardo "leggero" (massa base)
            self.model.body_mass[1] = base_torso_mass
            
        elif domain == "target":
            # Il target è il ghepardo "pesante" (+1.5kg rispetto al base)
            self.model.body_mass[1] = base_torso_mass + 1.5
            
        elif domain == "shift":
            # Per i grafici di robustezza: base + valore variabile
            self.model.body_mass[1] = base_torso_mass + mass_shift

        # Sicurezza: impedisci masse fisicamente impossibili
        self.model.body_mass[1] = max(0.01, self.model.body_mass[1])
        # Aggiorna la copia delle masse originali DOPO lo shift del dominio 
        # (così i wrapper di DR randomizzano attorno alla nuova massa specifica del dominio)
        self.original_masses = np.copy(self.model.body_mass)
    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        # z, angle = self.data.qpos[1:3]
        # state = self.state_vector()[2:]

        # min_state, max_state = self._healthy_state_range
        # min_z, max_z = self._healthy_z_range
        # min_angle, max_angle = self._healthy_angle_range

        # healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        # healthy_z = min_z < z < max_z
        # healthy_angle = min_angle < angle < max_angle

        # is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return True

    def _get_obs(self):
        """Costruisce il vettore delle osservazioni concatenando posizioni e velocità."""
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            # Escludiamo la coordinata x (solitamente la prima in qpos)
            position = position[1:]

        return np.concatenate((position, velocity)).astype(np.float64)

    def step(self, action):
        """Esegue un passo di simulazione, calcola reward e osservazione successiva."""
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        z, angle = self.data.qpos[1:3]
        #print(f"\nDEBUG: Altezza z={z:.3f}, Angolo={angle:.3f}") 
        #print(f"\nTutto qpos: {self.data.qpos}")
        x_position_after = self.data.qpos[0]
    
        x_velocity = (x_position_after - x_position_before) / self.dt
        
        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        # HalfCheetah di default non ha condizioni di terminazione (is_done sempre False)
        truncated = False

        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info
        }

        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def _get_rew(self, x_velocity: float, action):
        """Calcola la funzione di reward separata per facilitare il tuning."""
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
       
        rewards = forward_reward + healthy_reward
        ctrl_cost = self.control_cost(action)
        cost= ctrl_cost

        reward = rewards- cost
        
        reward_info = {
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }
        return reward, reward_info

    def _get_reset_info(self):
        """Restituisce informazioni reset)."""
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }
    def reset_model(self):
        """
        Resetta lo stato del robot. Viene chiamato automaticamente da env.reset().
        """
        # Definiamo l'entità del rumore per la posizione e la velocità
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # qpos: posizioni delle articolazioni
        # qvel: velocità delle articolazioni
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        # Applichiamo lo stato al simulatore MuJoCo
        self.set_state(qpos, qvel)

        # Restituiamo l'osservazione iniziale
        return self._get_obs()
    
    def set_random_parameters(self):
        """Applica parametri campionati casualmente al modello MuJoCo."""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Campiona le masse dei link (tranne il torso) per il Domain Randomization."""
        # HalfCheetah ha 8 corpi in totale (world + 7 segmenti). 
        # Escludendo world (0) e torso (1) come da tua nota, restano 6 link.
        n_links = len(self.model.body_mass) - 2
        
        # Esempio: variazione tra il 50% e il 150% della massa originale
        new_masses = self.original_masses[2:] * self.np_random.uniform(0.5, 1.5, size=n_links)
        return new_masses

    def get_parameters(self):
        """Restituisce le masse attuali dei link (escluso torso)."""
        return np.array(self.model.body_mass[2:]).copy()

    def set_parameters(self, task):
        """Imposta le masse dei link nel simulatore MuJoCo."""
        # masses deve avere dimensione (n_corpi - 2)
        self.model.body_mass[2:] = task

"""
    Registered environments
"""

gym.register(
    id="CustomHalfCheetah-v0",
    entry_point="%s:CustomHalfCheetah" % __name__,
    max_episode_steps=1000,
)

gym.register(
    id="CustomHalfCheetah-source-v0",
    entry_point="%s:CustomHalfCheetah" % __name__,
    max_episode_steps=1000,
    kwargs={"domain": "source"},
)

gym.register(
    id="CustomHalfCheetah-target-v0",
    entry_point="%s:CustomHalfCheetah" % __name__,
    max_episode_steps=1000,
    kwargs={"domain": "target"},
)

gym.register(
        id="CustomHalfCheetah-shift-v0",
        entry_point="%s:CustomHalfCheetah" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "shift"}
)