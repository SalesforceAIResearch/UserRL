from .config import IntentionGymConfig, get_default_config, get_demo_config
from .env.intention_env import IntentionEnv

__all__ = ["IntentionEnv", "IntentionGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='IntentionGym-v0', entry_point='intentiongym.env:IntentionEnv', max_episode_steps=20)
except ImportError:
    pass
