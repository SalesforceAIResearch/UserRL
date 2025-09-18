import os
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AlfworldGymConfig:
    """Complete configuration for AlfworldGym environment."""
    
    # Environment configuration
    env_config_path: str = "./alfworldgym/config/template.yaml"
    max_steps: int = 20
    verbose: bool = False
    seed: Optional[int] = 42

    # Alfworld-specific configuration
    env_type: str = "AlfredTWEnv"  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
    train_eval: str = "eval_in_distribution"  # 'train', 'eval_in_distribution', 'eval_out_of_distribution'
    batch_size: int = 1
    
    # Scoring configuration
    success_reward: float = 1.0
    failure_reward: float = 0.0
    step_penalty: float = 0.0
    normalize_rewards: bool = False
    
    # Data configuration - simplified for alfworld
    # Only support single mode since alfworld generates scenarios on reset
    data_mode: str = "single"  # Only "single" supported - each reset generates new scenario
    data_source: int = -1
    
    def __post_init__(self):
        # Validate configuration
        if self.data_mode != "single":
            raise ValueError("AlfworldGym only supports data_mode='single' since scenarios are generated on reset")
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'AlfworldGymConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AlfworldGymConfig':
        """Load configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        if self.success_reward < 0:
            raise ValueError("success_reward must be non-negative")
        
        if self.env_type not in ["AlfredTWEnv", "AlfredThorEnv", "AlfredHybrid"]:
            raise ValueError("env_type must be 'AlfredTWEnv', 'AlfredThorEnv', or 'AlfredHybrid'")
        
        if self.train_eval not in ["train", "eval_in_distribution", "eval_out_of_distribution"]:
            raise ValueError("train_eval must be 'train', 'eval_in_distribution', or 'eval_out_of_distribution'")
        
        if self.data_mode != "single":
            raise ValueError("data_mode must be 'single' for AlfworldGym")
        
        if self.batch_size != 1:
            raise ValueError("batch_size must be 1 for AlfworldGym")


def get_default_config() -> AlfworldGymConfig:
    """Get default configuration for AlfworldGym."""
    return AlfworldGymConfig()


def get_demo_config() -> AlfworldGymConfig:
    """Get demo configuration with verbose output enabled."""
    return AlfworldGymConfig(
        verbose=True,
        max_steps=30,
        success_reward=1.0,
        step_penalty=0.01
    ) 