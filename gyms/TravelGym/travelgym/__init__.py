from .config import TravelGymConfig, get_default_config, get_demo_config
from .env.travel_env import TravelEnv

__all__ = ["TravelEnv", "TravelGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='TravelGym-v0', entry_point='travelgym.env:TravelEnv', max_episode_steps=25)
except ImportError:
    pass
