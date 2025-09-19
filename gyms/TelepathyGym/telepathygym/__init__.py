from .config import TelepathyGymConfig, get_default_config, get_demo_config
from .env.telepathy_env import TelepathyEnv

__all__ = ["TelepathyEnv", "TelepathyGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='TelepathyGym-v0', entry_point='telepathygym.env:TelepathyEnv', max_episode_steps=20)
except ImportError:
    pass
