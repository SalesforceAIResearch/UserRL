from .config import TauGymConfig, get_default_config, get_demo_config
from .env.tau_env import TauEnv

__all__ = ["TauEnv", "TauGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='TauGym-v0', entry_point='taugym.env:TauEnv', max_episode_steps=30)
except ImportError:
    pass
