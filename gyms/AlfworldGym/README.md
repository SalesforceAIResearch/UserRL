# AlfworldGym

A Gymnasium-compatible environment for alfworld household task completion. This gym serves as a standardized proxy between reinforcement learning agents and the alfworld environment, providing a clean interface for research in household task automation and embodied AI.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Alfworld Integration**: Direct integration with the alfworld household simulation framework
- **Action Parsing**: Supports structured `[action]` and `[finish]` command formats
- **Async Support**: Both synchronous and asynchronous step methods for flexible usage
- **Configurable**: YAML-based configuration system with comprehensive customization options
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Data Preparation

To use AlfworldGym effectively, you need to set up the underlying alfworld environment and data:

### 1. Clone the Original Alfworld Repository

```bash
git clone https://github.com/alfworld/alfworld.git
cd alfworld
```

### 2. Install Alfworld Dependencies

Follow the installation instructions in the original alfworld repository to:
- Install all required dependencies
- Install the alfworld package itself

### 3. Download Required Data and Models

Execute the alfworld data download script:

```bash
# From the alfworld repository root
./scripts/alfworld-download
```

This downloads necessary game data and pre-trained models.

### 4. Configure Data Paths

You have two options for data placement:

**Option A (Recommended)**: Keep data in the original alfworld location and update paths in the configuration file.

**Option B (Optional)**: Copy the downloaded data and models to `alfworldgym/data/` in this repository.

### 5. Update Configuration Template

Edit the configuration file at `alfworldgym/config/template.yaml`:

1. **Update all `PROJECT_ROOT` placeholders** with the absolute path to your AlfworldGym installation
2. **Verify data paths** point to the correct locations of your downloaded alfworld data
3. **Ensure model paths** are correctly specified for your system

Example path updates:
```yaml
# Before
data_path: 'PROJECT_ROOT/AlfworldGym/alfworldgym/data/json_2.1.1/train'

# After (adjust path to your installation)
data_path: '/path/to/your/AlfworldGym/alfworldgym/data/json_2.1.1/train'
```

## Quick Start

```python
import alfworldgym
from alfworldgym import AlfworldEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = AlfworldEnv(config)

# Reset to get initial task
observation, info = env.reset()
print(f"Task: {observation['task']}")
print(f"Initial state: {observation['feedback']}")

# Take actions in the environment
obs, reward, terminated, truncated, info = env.step("[action] look")
print(f"Observation: {obs['feedback']}")

obs, reward, terminated, truncated, info = env.step("[action] inventory")
print(f"Inventory: {obs['feedback']}")

# Finish episode
obs, reward, terminated, truncated, info = env.step("[finish]")
print(f"Final reward: {reward}")

env.close()
```

## Action Format

The environment accepts actions in two specific formats:

### 1. Household Actions: `[action] <command>`

Execute specific household tasks and interactions:

- **Navigation**: `[action] go to kitchen`, `[action] go to living room`
- **Object Manipulation**: `[action] take apple from counter`, `[action] put apple in fridge`
- **Container Interaction**: `[action] open fridge`, `[action] close drawer`
- **Examination**: `[action] look`, `[action] inventory`, `[action] examine fridge`

### 2. Episode Termination: `[finish]`

End the current episode:
- Use when the task is completed successfully
- Use when you want to give up on the current task

## Configuration

### Basic Configuration

```python
from alfworldgym import AlfworldGymConfig

config = AlfworldGymConfig(
    max_steps=50,
    verbose=True,
    env_type="AlfredTWEnv",  # Environment backend
    train_eval="eval_in_distribution",
    success_reward=1.0,
    step_penalty=0.01
)

env = AlfworldEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum steps per episode | `50` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `env_type` | Alfworld environment backend | `"AlfredTWEnv"` | `"AlfredTWEnv"`, `"AlfredThorEnv"`, `"AlfredHybrid"` |
| `train_eval` | Evaluation mode | `"eval_in_distribution"` | Various evaluation modes |
| `success_reward` | Reward for task completion | `1.0` | Any float |
| `failure_reward` | Reward for task failure | `0.0` | Any float |
| `step_penalty` | Penalty per step taken | `0.0` | Any float |
| `normalize_rewards` | Normalize rewards to [0,1] | `False` | `True`/`False` |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import alfworldgym

# Use the registered environment
env = gym.make('AlfworldGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new alfworld scenario and returns the initial observation
2. **Step**: Processes the agent's action, forwards it to alfworld, and returns the response
3. **Reward Calculation**: Assigns rewards based on task completion (1.0 for success, 0.0 for failure)
4. **Termination**: Episode ends when task is completed or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without completion

## Example Actions

### Complete Action Examples

```python
# Navigation commands
env.step("[action] go to kitchen")
env.step("[action] go to living room")
env.step("[action] go to bedroom")

# Object interaction
env.step("[action] take apple from counter")
env.step("[action] put apple in fridge")
env.step("[action] pick up book from table")
env.step("[action] place book on shelf")

# Container and appliance interaction
env.step("[action] open fridge")
env.step("[action] close fridge")
env.step("[action] open drawer")
env.step("[action] turn on microwave")

# Examination and information gathering
env.step("[action] look")
env.step("[action] inventory")
env.step("[action] examine fridge")
env.step("[action] look at counter")

# Task completion
env.step("[finish]")
```

## Async Usage

For applications requiring asynchronous operation:

```python
import asyncio

async def run_async_example():
    env = AlfworldEnv()
    obs, info = env.reset()
    
    # Use async step method for non-blocking operation
    obs, reward, terminated, truncated, info = await env.step_async("[action] look")
    print(f"Async observation: {obs['feedback']}")
    
    env.close()

# Run the async example
asyncio.run(run_async_example())
```

