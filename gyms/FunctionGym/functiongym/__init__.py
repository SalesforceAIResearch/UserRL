"""
FunctionGym: A mathematical function learning environment for reinforcement learning.

This package provides a Gymnasium-compatible environment where agents can learn
hidden mathematical rules by testing different number combinations and receiving feedback.
"""

from .config import FunctionGymConfig, get_default_config, get_demo_config
from .env.function_env import FunctionEnv

__version__ = "1.0.0"
__author__ = "FunctionGym Team"

__all__ = [
    "FunctionEnv",
    "FunctionGymConfig", 
    "get_default_config",
    "get_demo_config"
]

# Register the environment with Gymnasium
try:
    import gymnasium as gym
    gym.register(
        id='FunctionGym-v0',
        entry_point='functiongym.env:FunctionEnv',
        max_episode_steps=20,
    )
except ImportError:
    # Gymnasium not available, skip registration
    pass 