"""
IntentionGym: An intention guessing simulation environment for reinforcement learning.

This package provides a Gymnasium-compatible environment where agents attempt
to clarify vague user tasks by asking targeted questions to uncover missing details
through multi-round conversations.
"""

from .config import IntentionGymConfig, get_default_config, get_demo_config
from .env.intention_env import IntentionEnv

__version__ = "1.0.0"
__author__ = "IntentionGym Team"

__all__ = [
    "IntentionEnv",
    "IntentionGymConfig", 
    "get_default_config",
    "get_demo_config",
]

# Register the environment with Gymnasium
try:
    import gymnasium as gym
    gym.register(
        id='IntentionGym-v0',
        entry_point='intentiongym.env:IntentionEnv',
        max_episode_steps=20,
    )
except ImportError:
    # Gymnasium not available, skip registration
    pass 