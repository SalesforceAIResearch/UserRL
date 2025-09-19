import os
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class IntentionGymConfig:
    """Complete configuration for IntentionGym environment."""
    
    # Model configuration
    api_key: str = ""
    model_name: str = "gpt-4o"
    base_url: str = ""  # Custom base URL for the API endpoint
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 10
    
    # Environment configuration
    max_steps: int = 20
    verbose: bool = False
    seed: Optional[int] = 42
    
    # Scoring configuration
    reward_scale: float = 1.0
    step_penalty: float = 0.0
    normalize_rewards: bool = True
    multi_detail_penalty: float = 0.2  # Penalty for covering multiple details in one question
    
    # Data configuration
    data_mode: str = "random"  # "random", "single", "list"
    data_source: Optional[Union[str, List[str]]] = None  # task ID(s) or None for random
    
    def __post_init__(self):
        # Auto-load API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'IntentionGymConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IntentionGymConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable "
                "or provide it in the config."
            )
        
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError("data_mode must be 'random', 'single', or 'list'")
        
        if self.data_mode == "single" and not isinstance(self.data_source, str):
            raise ValueError("data_mode 'single' requires data_source to be a string")
        
        if self.data_mode == "list" and not isinstance(self.data_source, list):
            raise ValueError("data_mode 'list' requires data_source to be a list")
        
        return True


# Pre-built configurations
def get_default_config() -> IntentionGymConfig:
    """Get default configuration."""
    return IntentionGymConfig()


def get_demo_config() -> IntentionGymConfig:
    """Get configuration optimized for demos."""
    return IntentionGymConfig(
        max_steps=15,
        verbose=True,
        step_penalty=0.01,
        multi_detail_penalty=0.3
    ) 