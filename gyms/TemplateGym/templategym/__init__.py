from .config import TemplateGymConfig, get_default_config, get_demo_config
from .env.template_env import TemplateEnv

__all__ = ["TemplateEnv", "TemplateGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='TemplateGym-v0', entry_point='templategym.env:TemplateEnv', max_episode_steps=20)
except ImportError:
    pass
