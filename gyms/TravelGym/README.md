# TravelGym

A Gymnasium-compatible environment for travel planning preference elicitation simulation through reinforcement learning. This gym provides a standardized interface for agents to interact with simulated users to elicit travel preferences through natural conversation, function calls, and targeted recommendations, enabling research in conversational AI and preference elicitation strategies.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Travel Planning Simulation**: Agents help users plan trips by eliciting preferences
- **Multi-Modal Actions**: Supports conversation, function calls, and recommendations
- **Preference Elicitation**: Agents learn to extract user preferences through strategic questioning
- **Function Call Integration**: Agents can search for travel options using structured function calls
- **Configurable**: Flexible configuration system for different travel scenarios
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY as environment variable.

```python
import travelgym
from travelgym import TravelEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = TravelEnv(config)

# Reset to get initial travel scenario
observation, info = env.reset()
print(f"User: {observation['feedback']}")

# Ask about travel preferences
obs, reward, terminated, truncated, info = env.step("[action] What destination are you traveling to?")
print(f"User: {obs['feedback']}")
print(f"Reward: {reward}")

# Search for hotel options
obs, reward, terminated, truncated, info = env.step('[search] {"function_name": "hotel", "arguments": {"location": "Los Angeles", "room_type": "Queen"}}')
print(f"Search results: {obs['feedback']}")

# Make a recommendation
obs, reward, terminated, truncated, info = env.step("[answer] H13")
print(f"Result: {obs['feedback']}")
print(f"Final reward: {reward}")

env.close()
```

## Action Format

The environment accepts actions in four specific formats:

### 1. Conversation: `[action] <question>`

Ask questions to elicit user preferences:

- **Example**: `[action] What destination are you traveling to?` - Ask about travel destination
- **Example**: `[action] Do you have any preferences for your hotel room?` - Ask about hotel preferences
- **Example**: `[action] What's your budget for this trip?` - Ask about budget constraints
- **Purpose**: Elicit user preferences through natural conversation

### 2. Function Calls: `[search] <json_function_call>`

Search for travel options using structured function calls:

- **Example**: `[search] {"function_name": "hotel", "arguments": {"location": "Los Angeles", "room_type": "Queen"}}` - Search for hotels
- **Example**: `[search] {"function_name": "flight", "arguments": {"origin": "NYC", "destination": "LAX", "date": "2024-01-15"}}` - Search for flights
- **Example**: `[search] {"function_name": "apartment", "arguments": {"location": "Downtown", "capacity": 4}}` - Search for apartments
- **Purpose**: Retrieve travel options based on user preferences

### 3. Recommendations: `[answer] <option_id>`

Recommend specific travel options by ID:

- **Example**: `[answer] H13` - Recommend hotel option H13
- **Example**: `[answer] F5` - Recommend flight option F5
- **Example**: `[answer] A2` - Recommend apartment option A2
- **Purpose**: Provide final recommendations to complete the travel planning

### 4. Episode Termination: `[finish]`

End the current episode:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the episode when travel planning is complete

## Configuration

### Basic Configuration

```python
from travelgym import TravelGymConfig

config = TravelGymConfig(
    max_steps=20,
    verbose=True,
    data_mode="random",
    reward_scale=1.0,
    step_penalty=0.0,
    normalize_rewards=False,
    search_correct_reward=0.2,
    preference_correct_reward=0.2,
    choice_best_reward=1.0,
    choice_correct_reward=0.8
)

env = TravelEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum steps per episode | `20` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `data_mode` | Scenario selection mode | `"random"` | `"random"`, `"single"`, `"list"` |
| `data_source` | Specific scenario to use | `"random"` | Scenario key string or list |
| `reward_scale` | Scale factor for rewards | `1.0` | Any positive float |
| `step_penalty` | Penalty per step taken | `0.0` | Any float |
| `normalize_rewards` | Normalize rewards to [0,1] | `False` | `True`/`False` |
| `search_correct_reward` | Reward for correct function calls | `0.2` | Any float |
| `preference_correct_reward` | Reward for eliciting preferences | `0.2` | Any float |
| `choice_best_reward` | Reward for best recommendations | `1.0` | Any float |
| `choice_correct_reward` | Reward for correct recommendations | `0.8` | Any float |
| `search_failure_interval` | Simulate search errors every N calls | `5` | Any positive integer |
| `elicitation_interval` | Proactive preference reveal interval | `3` | Any positive integer |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import travelgym

# Use the registered environment
env = gym.make('TravelGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new travel scenario and returns the initial observation
2. **Step**: Processes the agent's action (conversation, function call, or recommendation), evaluates with LLM, and returns feedback
3. **Reward Calculation**: Assigns rewards based on preference elicitation, function call correctness, and recommendation quality
4. **Termination**: Episode ends when correct recommendations are made or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without completion

## Environment Behavior

### Travel Planning Process

1. **Scenario Initialization**: Each episode contains a travel planning scenario with hidden user preferences
2. **Preference Elicitation**: Agents ask questions to discover user preferences for hotels, flights, apartments, restaurants, and rental cars
3. **Function Call Execution**: Agents search for travel options using structured function calls
4. **Recommendation Phase**: Agents provide specific recommendations by option ID
5. **Success Condition**: Episode succeeds when appropriate recommendations are made

### Available Travel Aspects

The environment supports comprehensive travel planning across multiple categories:

- **Hotels (H)**: Room types, amenities, ratings, costs
- **Flights (F)**: Routes, airlines, layovers, costs
- **Apartments (A)**: Room configurations, capacity, amenities
- **Restaurants (R)**: Cuisine types, ratings, price levels
- **Rental Cars (C)**: Brands, models, features, costs

### Reward System

- **Preference Elicitation**: **+0.2** reward for successfully eliciting user preferences
- **Correct Function Calls**: **+0.2** reward for valid search requests
- **Best Recommendations**: **+1.0** reward for recommending the optimal option
- **Correct Recommendations**: **+0.8** reward for recommending acceptable options
- **Step Penalty**: Configurable penalty per step taken
- **Search Failures**: Periodic simulated system errors for robustness testing

### User Simulation Features

- **Implicit Preference Revealing**: Users reveal preferences naturally through conversation
- **Proactive Elicitation**: Users may proactively share preferences if conversation goes off-topic
- **Realistic Responses**: GPT-4o simulates realistic user behavior and preferences
- **Preference Tracking**: System tracks which preferences have been elicited

## Example Actions

### Complete Action Examples

```python
# Elicit travel preferences
env.step("[action] What destination are you traveling to?")
env.step("[action] Do you have any hotel preferences?")
env.step("[action] What's your budget for accommodations?")

# Search for travel options
env.step('[search] {"function_name": "hotel", "arguments": {"location": "Los Angeles", "room_type": "Queen"}}')
env.step('[search] {"function_name": "flight", "arguments": {"origin": "NYC", "destination": "LAX"}}')
env.step('[search] {"function_name": "apartment", "arguments": {"location": "Downtown", "capacity": 4}}')

# Make recommendations
env.step("[answer] H13")  # Hotel recommendation
env.step("[answer] F5")   # Flight recommendation
env.step("[answer] A2")   # Apartment recommendation

# End episode
env.step("[finish]")
```

### Effective Travel Planning Strategy

```python
# Start with broad questions
env.step("[action] What destination are you traveling to?")
env.step("[action] When are you planning to travel?")

# Elicit specific preferences
env.step("[action] Do you have any preferences for your hotel room?")
env.step("[action] What type of accommodation do you prefer?")

# Search based on preferences
env.step('[search] {"function_name": "hotel", "arguments": {"location": "Los Angeles", "room_type": "Queen"}}')

# Make informed recommendations
env.step("[answer] H13")

# Or end if planning is complete
env.step("[finish]")
```

## API Requirements

### Required Environment Variables

- **`OPENAI_API_KEY`**: Required for LLM-based user simulation and evaluation

### Getting API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

### Setup Example

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

