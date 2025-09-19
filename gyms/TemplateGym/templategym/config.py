import os
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

@dataclass
class TemplateGymConfig:
    """Configuration class for TemplateGym environment."""
    
    # Model configuration
    api_key: str = ""
    model_name: str = "gpt-4o"
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 10
    
    # Environment configuration
    max_steps: int = 20
    verbose: bool = False
    seed: Optional[int] = None
    
    # Reward configuration
    reward_scale: float = 1.0
    step_penalty: float = 0.0
    normalize_rewards: bool = True
    
    # Data configuration
    data_mode: str = "random"  # "random", "single", "list"
    data_source: Optional[Union[str, List[str]]] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    def validate(self):
        """Validate configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError("data_mode must be 'random', 'single', or 'list'")
        return True

def get_default_config() -> TemplateGymConfig:
    """Get default configuration."""
    return TemplateGymConfig()

def get_demo_config() -> TemplateGymConfig:
    """Get configuration optimized for demos."""
    return TemplateGymConfig(
        verbose=True,
        max_steps=15,
        step_penalty=0.01,
    )
