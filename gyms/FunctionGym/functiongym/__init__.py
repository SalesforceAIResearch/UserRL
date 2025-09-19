from .config import FunctionGymConfig, get_default_config, get_demo_config
from .env.function_env import FunctionEnv

__all__ = ["FunctionEnv", "FunctionGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='FunctionGym-v0', entry_point='functiongym.env:FunctionEnv', max_episode_steps=20)
except ImportError:
    pass
