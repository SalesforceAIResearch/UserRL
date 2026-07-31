import os
from dataclasses import dataclass
from typing import Optional, Union, List


@dataclass
class SWEGymConfig:
    """Configuration class for SWEGym environment."""

    # Model configuration
    api_key: str = ""
    model_name: str = "gpt-4o"
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    # Coding instructions and rubrics are long, so the judge and user calls need
    # more headroom than the shorter conversational gyms.
    timeout: int = 60

    # Environment configuration
    max_steps: int = 20
    verbose: bool = False
    seed: Optional[int] = None

    # Reward configuration
    reward_scale: float = 1.0
    step_penalty: float = 0.0
    normalize_rewards: bool = False
    # A submitted change is accepted once the weighted rubric score reaches this.
    success_threshold: float = 0.85
    # Focused questions are worth more than ones that sweep up many requirements.
    multi_intent_penalty: float = 0.2
    # SWE-Together's "User Correction" axis: charge the agent when the user has
    # to push it back on track. Off by default so the reward is purely positive.
    correction_penalty: float = 0.0

    # Data configuration
    data_mode: str = "random"  # "random", "single", "list"
    data_source: Optional[Union[str, List[str]]] = None
    split: str = "test"  # "train" or "test"

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
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in (0, 1]")
        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError("data_mode must be 'random', 'single', or 'list'")
        if self.split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")
        return True


def get_default_config() -> SWEGymConfig:
    """Get default configuration."""
    return SWEGymConfig()


def get_demo_config() -> SWEGymConfig:
    """Get configuration optimized for demos."""
    return SWEGymConfig(
        verbose=True,
        max_steps=15,
    )
