from .config import TurtleGymConfig, get_default_config, get_demo_config
from .env.story_env import StoryEnv

__all__ = ["StoryEnv", "TurtleGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='TurtleSoup-v0', entry_point='turtlegym.env:StoryEnv', max_episode_steps=20)
except ImportError:
    pass
