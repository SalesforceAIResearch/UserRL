from .config import SearchGymConfig, get_default_config, get_demo_config
from .env.search_env import SearchEnv

__all__ = ["SearchEnv", "SearchGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='SearchGym-v0', entry_point='searchgym.env:SearchEnv', max_episode_steps=20)
except ImportError:
    pass
