from .config import PersuadeGymConfig, get_default_config, get_demo_config
from .env.persuade_env import PersuadeEnv

__all__ = ["PersuadeEnv", "PersuadeGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='PersuadeGym-v0', entry_point='persuadegym.env:PersuadeEnv', max_episode_steps=20)
except ImportError:
    pass
