# PersuadeGym

A Gymnasium-compatible environment for persuasion simulation through reinforcement learning. This gym provides a standardized interface for agents to persuade an AI environment to change its beliefs about various statements through multi-round conversations, enabling research in persuasive dialogue and belief change.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Persuasion Simulation**: Agents learn to change AI beliefs through strategic argumentation
- **Multi-round Conversations**: Supports interactive dialogue to shift stance positions
- **Delta-based Scoring**: Rewards based on stance changes with exponential scaling
- **Configurable**: Flexible configuration system for different persuasion scenarios
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY as environment variable.

```python
import persuadegym
from persuadegym import PersuadeEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = PersuadeEnv(config)

# Reset to get initial statement
observation, info = env.reset()
print(f"Statement: {observation['statement_text']}")
print(f"AI stance: {observation['current_stance']}")

# Make persuasion attempt
obs, reward, terminated, truncated, info = env.step("[action] Have you considered the economic implications?")
print(f"AI response: {obs['feedback']}")
print(f"New stance: {obs['current_stance']}")
print(f"Reward: {reward:.3f}")

# Finish episode
obs, reward, terminated, truncated, info = env.step("[finish]")
print(f"Final reward: {reward}")

env.close()
```

## Action Format

The environment accepts actions in three specific formats:

### 1. Persuasion Arguments: `[action] <argument>`

Make persuasion attempts to change the AI's stance:

- **Example**: `[action] Have you considered the economic implications?` - Present economic arguments
- **Example**: `[action] Studies show that this approach has negative long-term effects` - Use evidence-based arguments
- **Example**: `[action] What about the ethical concerns with this position?` - Raise ethical considerations
- **Purpose**: Present arguments to shift the AI's belief position

### 2. Alternative Arguments: `[answer] <argument>`

Same functionality as `[action]` - make any persuasion attempt:

- **Example**: `[answer] Consider the real-world consequences of this policy` - Present alternative arguments
- **Purpose**: Identical to `[action]` - provides flexibility in argument presentation

### 3. Episode Termination: `[finish]`

End the current conversation:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the conversation when satisfied with stance change

## Configuration

### Basic Configuration

```python
from persuadegym import PersuadeGymConfig

config = PersuadeGymConfig(
    max_steps=20,
    verbose=True,
    data_mode="random",
    step_penalty=0.0,
    normalize_rewards=True
)

env = PersuadeEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum arguments per episode | `20` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `data_mode` | Statement selection mode | `"random"` | `"random"`, `"single"`, `"list"` |
| `data_source` | Specific statement ID(s) to use | `None` | Statement ID string or list |
| `step_penalty` | Penalty per argument made | `0.0` | Any float |
| `normalize_rewards` | Normalize rewards to [0,1] | `True` | `True`/`False` |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import persuadegym

# Use the registered environment
env = gym.make('PersuadeGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new controversial statement and returns the initial observation
2. **Step**: Processes the agent's argument, evaluates stance change, and returns feedback
3. **Reward Calculation**: Assigns rewards based on stance change magnitude (delta-based exponential scaling)
4. **Termination**: Episode ends when AI reaches "Strongly Disagree" or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without reaching target stance

## Environment Behavior

### Persuasion Process

1. **Initial Stance**: AI starts with "Strongly Agree" on controversial statements
2. **Argument Evaluation**: Agents present arguments to shift the AI's position
3. **Stance Progression**: AI moves through stance levels based on argument quality
4. **Delta-based Rewards**: Higher rewards for larger stance changes
5. **Success Condition**: Episode succeeds when AI reaches "Strongly Disagree"

### Stance Progression System

- **Strongly Agree (1.0)** → **Agree (0.83)** → **Partly Agree (0.67)** → 
- **Neutral (0.5)** → **Partly Disagree (0.33)** → **Disagree (0.17)** → 
- **Strongly Disagree (0.0)** 🎯

### Reward System

- **1 level change**: **0.6** reward (e.g., Strongly Agree → Agree)
- **2 level change**: **0.8** reward (e.g., Strongly Agree → Partly Agree)
- **3 level change**: **0.9** reward (e.g., Strongly Agree → Neutral)
- **4+ level change**: **0.95-1.0** reward (major shifts)
- **No change or backward**: **0.0** reward

## Example Actions

### Complete Action Examples

```python
# Evidence-based arguments
env.step("[action] Studies show that this approach has negative long-term effects")
env.step("[action] Research indicates significant economic costs")
env.step("[action] Data from multiple countries supports this conclusion")

# Logical reasoning arguments
env.step("[action] This position creates logical inconsistencies")
env.step("[action] Consider the implications of this reasoning")
env.step("[action] This argument doesn't account for alternative explanations")

# Real-world example arguments
env.step("[action] Look at what happened in similar cases")
env.step("[action] Real-world implementation shows different results")
env.step("[action] Historical examples demonstrate the problems")

# Ethical and moral arguments
env.step("[action] What about the ethical implications?")
env.step("[action] This raises serious moral concerns")
env.step("[action] Consider the impact on vulnerable populations")

# Alternative format (same functionality)
env.step("[answer] Have you considered the unintended consequences?")
env.step("[answer] What about the long-term sustainability?")

# End conversation
env.step("[finish]")
```

### Effective Persuasion Strategy

```python
# Start with strong evidence-based arguments
env.step("[action] Multiple peer-reviewed studies contradict this position")
env.step("[action] The data shows significant negative outcomes")

# Follow with logical reasoning
env.step("[action] This creates logical inconsistencies in your argument")
env.step("[action] Consider the implications of this reasoning")

# Use real-world examples
env.step("[action] Real-world implementation shows different results")
env.step("[finish]")  # End when satisfied with stance change
```

