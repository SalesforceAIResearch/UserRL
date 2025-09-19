# UserRL Gyms Collection

A comprehensive collection of Gymnasium-compatible environments for reinforcement learning research across diverse domains. Each gym provides a standardized interface while implementing domain-specific logic for various research applications.

## Available Gyms

### 🏠 AlfworldGym
**Domain**: Household task completion simulation  
**Description**: Agents learn to complete household tasks in a simulated environment using natural language instructions.  
**Key Features**: Object manipulation, navigation, task planning, LLM-based evaluation  
**Action Types**: `[action]`, `[finish]`  
**Dependencies**: `alfworld` (install from source)

### 🔢 FunctionGym  
**Domain**: Mathematical function learning  
**Description**: Agents learn to understand and work with mathematical functions through exploration and problem-solving.  
**Key Features**: Function discovery, parameter learning, mathematical reasoning  
**Action Types**: `[action]`, `[search]`, `[answer]`, `[finish]`  
**Dependencies**: None (self-contained)

### 🎯 IntentionGym
**Domain**: AI intention guessing simulation  
**Description**: Agents learn to guess user intentions through strategic questioning and conversation.  
**Key Features**: Intention inference, conversational AI, preference elicitation  
**Action Types**: `[action]`, `[finish]`  
**Dependencies**: OpenAI API

### 💬 PersuadeGym
**Domain**: AI persuasion simulation  
**Description**: Agents learn persuasive communication strategies through simulated user interactions.  
**Key Features**: Persuasion tactics, argumentation, social influence  
**Action Types**: `[action]`, `[finish]`  
**Dependencies**: OpenAI API

### 🔍 SearchGym
**Domain**: Search-based question answering  
**Description**: Agents learn to answer questions by searching the web and synthesizing information.  
**Key Features**: Web search, information synthesis, fact verification  
**Action Types**: `[search]`, `[answer]`, `[finish]`  
**Dependencies**: OpenAI API, Serper API

### 🛠️ TauGym
**Domain**: Tool-agent-user interactions  
**Description**: Agents learn to use tools and interact with users in complex multi-agent scenarios.  
**Key Features**: Tool usage, multi-agent coordination, user interaction  
**Action Types**: `[search]`, `[action]`, `[answer]`, `[finish]`  
**Dependencies**: `tau-bench` (install from source)

### 🧠 TelepathyGym
**Domain**: Mind reading games  
**Description**: Agents learn to guess what entities an AI is thinking of through strategic yes/no questions.  
**Key Features**: Logical reasoning, strategic questioning, entity guessing  
**Action Types**: `[action]`, `[answer]`, `[finish]`  
**Dependencies**: OpenAI API

### 🧩 TurtleGym
**Domain**: Turtle Soup lateral thinking puzzles  
**Description**: Agents solve mysterious story scenarios by asking questions and providing explanations.  
**Key Features**: Lateral thinking, puzzle solving, creative reasoning  
**Action Types**: `[action]`, `[answer]`, `[finish]`  
**Dependencies**: OpenAI API

### ✈️ TravelGym
**Domain**: Travel planning preference elicitation  
**Description**: Agents learn to help users plan trips by eliciting preferences and making recommendations.  
**Key Features**: Preference elicitation, function calls, recommendation systems  
**Action Types**: `[action]`, `[search]`, `[answer]`, `[finish]`  
**Dependencies**: OpenAI API

## Common Patterns

All gyms follow these established patterns:

### Action Format Standards
- **`[action]`**: Questions, requests, or general actions
- **`[answer]`**: Solutions, explanations, or final responses  
- **`[finish]`**: Episode termination
- **Domain-specific**: Additional prefixes as needed (e.g., `[search]`, `[recommend]`)

### Configuration System
- Dataclass-based configuration with sensible defaults
- Environment variable support for API keys
- Pre-built configurations (`get_default_config`, `get_demo_config`)
- Validation methods for parameter checking

### Reward Mechanisms
- **Delta-based scoring**: Rewards for improvement over previous attempts
- **Success thresholds**: Clear criteria for episode completion
- **Step penalties**: Encouraging efficiency
- **Multi-criteria evaluation**: Different aspects weighted appropriately

### LLM Integration
- **OpenAI API**: Standard integration pattern across gyms
- **Error handling**: Graceful fallbacks for API failures
- **Evaluation**: Using LLMs to assess agent performance
- **Simulation**: Using LLMs to simulate user/system behavior

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key (for most gyms)
- Additional API keys as specified per gym

### Installation
Each gym can be installed independently:

```bash
cd YourGym
pip install -e .
```

### Basic Usage
```python
import yourgym
from yourgym import YourEnv, get_default_config

# Set up API key
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Create and use environment
config = get_default_config()
env = YourEnv(config)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step("[action] your action")
env.close()
```

## Contributing New Gyms

We welcome contributions of new gyms! See [CONTRIBUTING.md](CONTRIBUTING.md) for comprehensive guidelines and use the [TemplateGym](TemplateGym/) as a starting point.

### Quick Start for Contributors
1. Copy `TemplateGym/` to create your new gym
2. Replace template content with your domain logic
3. Follow the established patterns and conventions
4. Add comprehensive tests and documentation
5. Submit a pull request

## Research Applications

These gyms enable research in:

- **Conversational AI**: IntentionGym, PersuadeGym, TelepathyGym
- **Tool Usage**: TauGym, SearchGym, TravelGym
- **Reasoning**: FunctionGym, TurtleGym, TelepathyGym
- **Planning**: AlfworldGym, TravelGym
- **Preference Learning**: IntentionGym, TravelGym
- **Multi-modal Interaction**: TravelGym, TauGym
