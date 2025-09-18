"""
AlfworldGym: An alfworld-based environment for reinforcement learning.

This package provides a Gymnasium-compatible environment where agents can interact
with alfworld environments, performing household tasks and receiving rewards based on task completion.
"""

from .config import AlfworldGymConfig, get_default_config, get_demo_config
from .env.alfworld_env import AlfworldEnv

__version__ = "1.0.0"
__author__ = "AlfworldGym Team"

__all__ = [
    "AlfworldEnv",
    "AlfworldGymConfig", 
    "get_default_config",
    "get_demo_config"
]

# Register the environment with Gymnasium
try:
    import gymnasium as gym
    gym.register(
        id='AlfworldGym-v0',
        entry_point='alfworldgym.env:AlfworldEnv',
        max_episode_steps=50,
    )
except ImportError:
    # Gymnasium not available, skip registration
    pass 