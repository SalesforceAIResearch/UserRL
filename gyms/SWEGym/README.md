# SWEGym

A Gymnasium environment where the agent plays the coding agent in a real software engineering session, working with a simulated user who knows what they want but does not say it all up front. Tasks, user scripts, and grading rubrics come from **[SWE-Together](https://arxiv.org/pdf/2606.29957)** (Wu et al., 2026), a benchmark of 109 reconstructed user/coding-agent sessions.

## Features

- **SWE-Together tasks**: 109 sessions reconstructed from real user/coding-agent transcripts (80 train, 29 test)
- **Reactive simulated user**: an LLM plays the user, answering questions, adding requirements, and pushing back when the agent drifts
- **Requirement elicitation reward**: credit for drawing out what the user actually wanted, weighted by how much it mattered
- **Rubric-graded submissions**: each task carries a frozen set of weighted completeness goals that sum to 1.0
- **Async support**: both `step()` and `step_async()` for concurrent rollouts

## Installation

```bash
cd gyms/SWEGym
pip install -e .
```

## Quick Start

```python
import swegym

# Create environment with default configuration
config = swegym.get_default_config()
config.data_mode = "single"
config.data_source = "triton-msvc-c4267-warnings"
env = swegym.SWEEnv(config=config)

# Reset to get the user's opening message
observation, info = env.reset()
print(observation["instruction"])

# Ask the user something the opening message left unsaid
observation, reward, terminated, truncated, info = env.step(
    "[action] What is the signature of op->getResult in this codebase?"
)
print(observation["feedback"], reward)

# Submit the change you would make
observation, reward, terminated, truncated, info = env.step(
    "[answer] Cast the loop index to unsigned at both template instantiation sites."
)
print(observation["best_score"])
```

## Action Format

The agent interacts through two operations plus termination. **There is no `search` operation in this environment**; a `search` action is rejected as malformed.

### 1. Talking to the user: `[action] <message>`

Ask a clarifying question, confirm an assumption, or state the approach you are about to take. The simulated user replies in character.

Reward comes from *intent coverage*: each task carries the requirements the real user expressed during the original session, and a message earns credit when it surfaces one of them. Credit is set by the most important intent the message elicited (1.0, 0.7, or 0.4 for importance 3, 2, 1) and is reduced by `multi_intent_penalty` for each extra intent beyond the first, so focused questions beat sweeping ones. Each intent pays out only once.

### 2. Submitting a change: `[answer] <description of the change>`

Describe the concrete code change: which files and functions you would modify and what the modification is. A judge scores it against the task's frozen completeness goals, giving each goal 0, 0.5, or 1.0 and combining them with the goal weights into a score in `[0, 1]`.

Reward is the **improvement over your best previous submission**, so resubmitting a worse change costs nothing and refining a good one pays the difference. The episode terminates once a submission reaches `success_threshold`.

### 3. Episode Termination: `[finish]`

Ends the session with zero reward and reports final intent coverage and best score.

## Configuration

### Basic Configuration

```python
import swegym

config = swegym.get_default_config()
config.model_name = "gpt-4o"       # model behind the simulated user and the judge
config.max_steps = 20
config.success_threshold = 0.85    # submission score that ends the episode
config.split = "test"              # which split "random" mode samples from
```

### Configuration Options

| Option | Default | Description |
|---|---|---|
| `model_name` | `gpt-4o` | Model used for the simulated user, the coverage evaluator, and the judge |
| `max_tokens` | `2048` | Token budget per LLM call; the judge needs room for one entry per goal |
| `timeout` | `60` | Seconds per LLM call. Higher than the other gyms because instructions and rubrics are long |
| `max_steps` | `20` | Turn budget |
| `success_threshold` | `0.85` | Submission score that terminates the episode, matching the source benchmark's bar |
| `multi_intent_penalty` | `0.2` | Discount per extra intent elicited by a single message |
| `correction_penalty` | `0.0` | Charge applied when the user has to correct the agent (see below) |
| `data_mode` | `random` | `random`, `single`, or `list` |
| `data_source` | `None` | Task id, or list of task ids |
| `split` | `test` | `train` or `test`, used when `data_mode="random"` |

The `correction_penalty` knob exposes the source benchmark's second axis, which measures how much the user had to push the agent back on track. It is `0.0` by default so that reward is purely positive; set it above zero to penalise turns where the simulated user replies with a correction. The per-episode correction count is always reported in `info["n_corrections"]` regardless.

## Gymnasium Registration

```python
import gymnasium as gym
import swegym

# Use the registered environment
env = gym.make('SWEGym-v0')
```

## Data Flow

```
SWE-Together session
   -> instruction.md              -> the user's opening message (shown to the agent)
   -> user_simulation_prompt.md   -> the simulated user's private script
   -> oracle_intents.json         -> requirements the real user expressed (coverage reward)
   -> canonical_goals.json        -> weighted completeness rubric (submission reward)
```

The three LLM roles are kept apart on purpose. The simulated user sees its script but not the rubric, so it cannot leak the grading key back to the agent. The coverage evaluator sees the intents but never writes a reply. The judge sees the rubric but not the user's script.

## Environment Behavior

### Session Process

1. The agent receives the user's opening message, which is genuinely underspecified
2. The agent alternates between asking the user things and submitting changes
3. Intents pay out the first time the agent surfaces them
4. Submissions are scored against the rubric, and only improvements pay
5. The episode ends on a submission at or above `success_threshold`, on `[finish]`, or when the turn budget runs out

### Task Collection

Tasks come from the SWE-Together benchmark: real sessions across 109 repositories, spanning build errors, refactors, feature integration, and bug fixes. Instructions are verbatim user messages, so they range from a one-line error paste to a long log dump.

### Reward System

| Event | Reward |
|---|---|
| Message elicits intents | `importance_reward - multi_intent_penalty * (n_elicited - 1)` |
| Message elicits nothing | `0.0` |
| User replies with a correction | `- correction_penalty` |
| Submission scores above your best | `new_score - best_score` |
| Submission does not improve | `0.0` |
| `[finish]` or malformed action | `0.0` |

### LLM Failure Handling

When an LLM call fails after its retries, the turn reports `observation["llm_error"] = True` and `info["llm_success"] = False`. A failed judge call does **not** overwrite the best score or count as a zero-scoring submission, so an outage cannot be mistaken for a wrong answer.

## Example Actions

### Complete Action Examples

```python
# Ask about an interface the instruction assumed you knew
env.step("[action] What is the signature of op->getResult in this codebase?")

# Check the scope of the fix
env.step("[action] Are there other narrowing conversion sites in this file, or just the one in the error?")

# Confirm an approach before committing to it
env.step("[action] I plan to cast at the lambda capture rather than change the API. Does that work for you?")

# Submit a change
env.step("[answer] In WarpSpecializeUtility.cpp, capture the loop index as idx = static_cast<unsigned>(idx) in the partition->walk lambda, and apply the same cast at the second instantiation site.")

# End the session
env.step("[finish]")
```

### Effective Strategy

```python
# Establish the constraints the opening message left out
env.step("[action] Which compiler and flags are you building with?")

# Probe the scope before writing anything
env.step("[action] Should this fix only the reported site, or every similar site in the file?")

# Submit early, then refine against the feedback
env.step("[answer] <first attempt>")
env.step("[action] Is there anything about my approach you would change?")
env.step("[answer] <refined attempt>")
```

## API Requirements

### Required Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
```

### Setup Example

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

import swegym
env = swegym.SWEEnv(config=swegym.get_default_config())
```

Any OpenAI-compatible endpoint works, including a locally served model.

## Scope

This gym models the **interaction** half of a coding session: eliciting requirements from the user and describing the change. It does not execute code, so a submission is graded on the change it describes rather than on a test suite. The source benchmark runs its agents in a sandboxed repository and grades the resulting patch; if you need that, run SWE-Together directly.

## Source and Citation

All task content in this gym is derived from **SWE-Together** ([paper](https://arxiv.org/pdf/2606.29957), [website](https://togetherbench.com)). Each task's opening message, the simulated user's private script, the oracle intents, and the completeness rubric are taken from that benchmark's `tasks/<name>/` directory. Regenerate the dataset with the exporter shipped in the SWE-Together repository:

```bash
python export_userrl_dataset.py --out-dir <UserRL>/gyms/SWEGym/swegym/data
```

If you use this gym, please cite SWE-Together alongside UserRL:

```bibtex
@article{wu2026swetogether,
  title   = {SWE-Together: Evaluating Coding Agents in Interactive User Sessions},
  author  = {Wu, Yifan and Zhao, Zhuokai and Li, Songlin and Lee, Ho Hin and Zhu, Jiacheng
             and Wu, Shirley and Yu, Tianhe and Li, Serena and Zhang, Lizhu
             and Fan, Xiangjun and Li, Shengzhi},
  year    = {2026},
  journal = {arXiv preprint arXiv:2606.29957},
  url     = {https://arxiv.org/pdf/2606.29957}
}
```
