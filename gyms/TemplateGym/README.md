# TemplateGym

A Gymnasium-compatible environment template for [your domain] through reinforcement learning.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Async Support**: Both sync and async step methods
- **Configurable**: Flexible configuration system
- **Standard Actions**: `[action]`, `[answer]`, `[finish]` format

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import templategym
from templategym import TemplateEnv, get_default_config

# Create environment
config = get_default_config()
env = TemplateEnv(config)

# Reset environment
observation, info = env.reset()
print(f"Description: {observation['description']}")

# Take actions
obs, reward, terminated, truncated, info = env.step("[action] test")
print(f"Feedback: {obs['feedback']}")

obs, reward, terminated, truncated, info = env.step("[answer] template answer")
print(f"Score: {obs['current_score']}")

env.close()
```

## Action Format

- **`[action] <text>`**: Ask questions or take actions
- **`[answer] <text>`**: Provide solutions or answers  
- **`[finish]`**: End the episode

## Configuration

```python
from templategym import TemplateGymConfig

config = TemplateGymConfig(
    max_steps=20,
    verbose=True,
    data_mode="random",
    reward_scale=1.0,
    step_penalty=0.0,
    normalize_rewards=True
)
```

## Async Usage

```python
import asyncio

async def async_example():
    env = TemplateEnv()
    obs, info = await env.reset()
    
    obs, reward, terminated, truncated, info = await env.step_async("[action] test")
    print(f"Feedback: {obs['feedback']}")
    
    env.close()

asyncio.run(async_example())
```

## Customization

Replace the template logic with your domain-specific implementation:

1. **`_handle_action()`**: Process action commands
2. **`_handle_answer()`**: Evaluate answers
3. **`_evaluate_answer()`**: Scoring logic
4. **`_load_data()`**: Data loading
5. **Data files**: Add your domain data

## API Requirements

Set your OpenAI API key if using LLM features:

```bash
export OPENAI_API_KEY="your-key-here"
```
