from .config import AlfworldGymConfig, get_default_config, get_demo_config
from .env.alfworld_env import AlfworldEnv

__all__ = ["AlfworldEnv", "AlfworldGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='AlfworldGym-v0', entry_point='alfworldgym.env:AlfworldEnv', max_episode_steps=50)
except ImportError:
    pass
