"""SWEGym: user/coding-agent interaction sessions.

Tasks are derived from SWE-Together (https://arxiv.org/pdf/2606.29957).
"""

from .config import SWEGymConfig, get_default_config, get_demo_config
from .env.swe_env import SWEEnv

__all__ = ["SWEEnv", "SWEGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='SWEGym-v0', entry_point='swegym.env:SWEEnv', max_episode_steps=20)
except ImportError:
    pass
