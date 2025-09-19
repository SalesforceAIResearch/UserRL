# TelepathyGym

A Gymnasium-compatible environment for mind reading games through reinforcement learning. This gym provides a standardized interface for agents to guess what entity an AI is thinking of through strategic yes/no questions, enabling research in question-asking strategies and logical reasoning.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Mind Reading Game**: Agents learn to guess entities through strategic questioning
- **Yes/No Questions**: Supports binary question-answer interactions
- **Entity Guessing**: Agents make final guesses to complete the game
- **Configurable**: Flexible configuration system for different game scenarios
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY as environment variable.

```python
import telepathygym
from telepathygym import TelepathyEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = TelepathyEnv(config)

# Reset to get initial entity
observation, info = env.reset()
print(f"Goal: {observation['goal']}")

# Ask yes/no questions
obs, reward, terminated, truncated, info = env.step("[action] Is it alive?")
print(f"AI response: {obs['feedback']}")

obs, reward, terminated, truncated, info = env.step("[action] Is it a person?")
print(f"AI response: {obs['feedback']}")

# Make final guess
obs, reward, terminated, truncated, info = env.step("[answer] Albert Einstein")
print(f"Result: {obs['feedback']}")
print(f"Score: {obs['current_score']}")

# Finish episode
obs, reward, terminated, truncated, info = env.step("[finish]")
print(f"Final reward: {reward}")

env.close()
```

## Action Format

The environment accepts actions in three specific formats:

### 1. Yes/No Questions: `[action] <question>`

Ask binary questions to narrow down the entity:

- **Example**: `[action] Is it alive?` - Ask if the entity is living
- **Example**: `[action] Is it a person?` - Ask if the entity is a person
- **Example**: `[action] Is it an animal?` - Ask if the entity is an animal
- **Purpose**: Gather information through strategic questioning

### 2. Final Guess: `[answer] <entity>`

Submit your final guess for the entity:

- **Example**: `[answer] Albert Einstein` - Guess a famous person
- **Example**: `[answer] Lion` - Guess an animal
- **Example**: `[answer] Piano` - Guess an object
- **Purpose**: Make the final guess to complete the game

### 3. Episode Termination: `[finish]`

End the current episode:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the episode when done guessing or want to give up

## Configuration

### Basic Configuration

```python
from telepathygym import TelepathyGymConfig

config = TelepathyGymConfig(
    max_steps=20,
    verbose=True,
    data_mode="random",
    reward_scale=1.0,
    step_penalty=0.0,
    normalize_rewards=True
)

env = TelepathyEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum questions per episode | `20` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `data_mode` | Entity selection mode | `"random"` | `"random"`, `"single"`, `"list"` |
| `data_source` | Specific entity name(s) to use | `None` | Entity name string or list |
| `reward_scale` | Scale factor for rewards | `1.0` | Any positive float |
| `step_penalty` | Penalty per question asked | `0.0` | Any float |
| `normalize_rewards` | Normalize rewards to [0,1] | `True` | `True`/`False` |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import telepathygym

# Use the registered environment
env = gym.make('TelepathyGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new entity and returns the initial observation
2. **Step**: Processes the agent's question or guess, evaluates with LLM, and returns feedback
3. **Reward Calculation**: Assigns rewards based on guess correctness (1.0 for correct, 0.0 for incorrect)
4. **Termination**: Episode ends when correct guess is made or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without correct guess

## Environment Behavior

### Mind Reading Game Process

1. **Entity Selection**: Each episode contains a hidden entity (person, animal, object, place)
2. **Strategic Questioning**: Agents ask yes/no questions to narrow down possibilities
3. **LLM Evaluation**: AI provides contextual "Yes"/"No"/"Maybe" responses
4. **Final Guessing**: Agents submit their best guess for the entity
5. **Success Condition**: Episode succeeds when the correct entity is guessed

### Available Entities

The environment includes diverse entities across categories:

- **People**: Albert Einstein, Harry Potter, Marie Curie
- **Animals**: Lion, Elephant, Dolphin
- **Objects**: Piano, Smartphone, Bicycle
- **Places**: Eiffel Tower, Mount Everest, Grand Canyon
- **And more**: 400+ diverse entities total

### Reward System

- **Correct Guess**: **1.0** reward when the entity is guessed correctly
- **Incorrect Guess**: **0.0** reward when the guess is wrong
- **Questions**: **0.0** reward for asking questions (information gathering)
- **Step Penalty**: Configurable penalty per question asked

## Example Actions

### Complete Action Examples

```python
# Ask about living things
env.step("[action] Is it alive?")
env.step("[action] Is it a person?")
env.step("[action] Is it an animal?")

# Ask about characteristics
env.step("[action] Is it famous?")
env.step("[action] Is it fictional?")
env.step("[action] Is it from history?")

# Ask about categories
env.step("[action] Is it an object?")
env.step("[action] Is it a place?")
env.step("[action] Is it something you can touch?")

# Make guesses
env.step("[answer] Albert Einstein")
env.step("[answer] Lion")
env.step("[answer] Piano")
env.step("[answer] Eiffel Tower")

# End episode
env.step("[finish]")
```

### Effective Questioning Strategy

```python
# Start with broad categories
env.step("[action] Is it alive?")
env.step("[action] Is it a person?")

# Narrow down based on responses
env.step("[action] Is it a scientist?")
env.step("[action] Is it from history?")

# Make educated guess
env.step("[answer] Albert Einstein")

# Or end if unsure
env.step("[finish]")
```

## API Requirements

### Required Environment Variables

- **`OPENAI_API_KEY`**: Required for LLM-based question evaluation

### Getting API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

### Setup Example

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

