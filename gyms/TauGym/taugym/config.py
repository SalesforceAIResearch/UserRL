import os
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TauGymConfig:
    """Complete configuration for TauGym environment."""
    
    # Tau-bench configuration
    task_split: str = "test"
    task_category: str = "retail"  # "retail" or "airline"
    data_source: Optional[str] = None  # Task ID for single mode (e.g., "retail-019", "airline-005")
    data_mode: str = "single"  # Only support single mode as requested
    
    # User simulation configuration
    user_strategy: str = "llm"
    user_model: str = "gpt-4o"
    user_provider: str = "openai"
    api_key: str = ""
    base_url: str = ""  # Custom base URL for the API endpoint

    # Environment configuration
    max_steps: int = 30
    verbose: bool = False
    seed: Optional[int] = 42
    
    # Scoring configuration
    normalize_rewards: bool = False
    
    def __post_init__(self):
        # Auto-load API key from environment if not provided
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
    
    def validate(self):
        """Validate configuration parameters."""
        if self.task_category not in ["retail", "airline"]:
            raise ValueError("task_category must be 'retail' or 'airline'")
        
        if self.data_mode != "single":
            raise ValueError("Only 'single' data mode is supported")
        
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'TauGymConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TauGymConfig':
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


def get_default_config() -> TauGymConfig:
    """Get default configuration for TauGym."""
    return TauGymConfig()


def get_demo_config() -> TauGymConfig:
    """Get demo configuration with verbose output enabled."""
    return TauGymConfig(
        verbose=True,
        task_category="retail",
        task_id="retail-019"
    )

