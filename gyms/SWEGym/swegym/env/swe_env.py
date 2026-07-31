import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import SWEGymConfig, get_default_config
from .prompts import (
    evaluate_intent_coverage,
    evaluate_intent_coverage_async,
    evaluate_submission,
    evaluate_submission_async,
    respond_to_agent,
    respond_to_agent_async,
)
from .task_data import get_task_by_id, load_tasks

# Base reward for eliciting an intent, keyed by the intent's importance tier.
IMPORTANCE_REWARDS = {3: 1.0, 2: 0.7, 1: 0.4}

GOAL_TEXT = (
    "Work with the user to fix the problem they reported. Use `action` to talk to "
    "them and draw out what they actually need, and `answer` to submit your change."
)


class SWEEnv(gym.Env):
    """
    A user/coding-agent interaction environment built from SWE-Together sessions.

    The agent is dropped into the opening message of a real coding session and
    has to work with a simulated user who knows what they want but does not
    volunteer it. Talking to the user pays off when it surfaces a requirement the
    real user expressed; submitting a change pays off when it satisfies the
    task's frozen completeness rubric.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[SWEGymConfig] = None):
        super().__init__()
        self.config = config or get_default_config()
        self.config.validate()

        self.tasks = self._load_tasks()
        if not self.tasks:
            raise ValueError("No tasks available for the requested data configuration")

        self._rng = random.Random(self.config.seed)
        self._task_index = 0

        # Action space: talk to the user, submit a change, or end the session
        self.action_space = spaces.Discrete(3)

        # Observation space: dictionary containing session state and feedback
        self.observation_space = spaces.Dict({
            "task_id": spaces.Text(max_length=256),
            "goal": spaces.Text(max_length=1024),
            "feedback": spaces.Text(max_length=8192),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "current_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "best_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "intents_elicited": spaces.Box(low=0, high=1024, shape=(), dtype=np.int32),
            "total_intents": spaces.Box(low=0, high=1024, shape=(), dtype=np.int32),
        })

        self.current_task: Dict[str, Any] = {}
        self.remaining_intents: List[Dict[str, Any]] = []
        self.elicited_intents: List[Dict[str, Any]] = []
        self.conversation_history: List[Tuple[str, str]] = []
        self.message_history: List[str] = []
        self.reaction_history: List[str] = []
        self.judge_history: List[Dict[str, Any]] = []
        self.step_count = 0
        self.total_reward = 0.0
        self.best_score = 0.0
        self.best_judge_response: Optional[Dict[str, Any]] = None
        self.episode_complete = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_tasks(self) -> List[Dict[str, Any]]:
        if self.config.data_mode == "random":
            return load_tasks(self.config.split)
        if self.config.data_mode == "single":
            return [get_task_by_id(self.config.data_source)]
        if self.config.data_mode == "list":
            sources = self.config.data_source or []
            if isinstance(sources, str):
                sources = [sources]
            return [get_task_by_id(task_id) for task_id in sources]
        raise ValueError(f"Unknown data_mode: {self.config.data_mode}")

    def _select_task(self) -> Dict[str, Any]:
        if self.config.data_mode == "random":
            return self._rng.choice(self.tasks)
        task = self.tasks[self._task_index % len(self.tasks)]
        self._task_index += 1
        return task

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Start a new session on a task."""
        if seed is not None:
            self._rng = random.Random(seed)

        self.current_task = self._select_task()
        self.remaining_intents = [dict(i) for i in self.current_task["oracle_intents"]]
        self.elicited_intents = []
        self.conversation_history = []
        self.message_history = []
        self.reaction_history = []
        self.judge_history = []
        self.step_count = 0
        self.total_reward = 0.0
        self.best_score = 0.0
        self.best_judge_response = None
        self.episode_complete = False

        observation = {
            "task_id": self.current_task["id"],
            "instruction": self.current_task["instruction"],
            "repo_url": self.current_task.get("repo_url", ""),
            "base_commit": self.current_task.get("base_commit", ""),
            "goal": GOAL_TEXT,
            "feedback": self.current_task["instruction"],
            "step_count": self.step_count,
            "episode_complete": False,
            "current_score": 0.0,
            "best_score": 0.0,
            "intents_elicited": 0,
            "total_intents": len(self.remaining_intents),
        }
        info = {
            "task_id": self.current_task["id"],
            "total_intents": len(self.remaining_intents),
            "total_goals": len(self.current_task["completeness_goals"]),
            "conversation_history": [],
        }
        if self.config.verbose:
            print(f"[SWEGym] Task {self.current_task['id']}")
        return observation, info

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        action_str, kind, payload = self._begin_step(action_input)
        if kind == "finish":
            return self._finish_step()
        if kind == "invalid":
            return self._invalid_step(action_str)

        if kind == "action":
            response, reaction, user_ok = respond_to_agent(
                payload, self.current_task, self._model_config(), self.conversation_history
            )
            elicited, coverage_ok = evaluate_intent_coverage(
                payload,
                self.current_task,
                self._model_config(),
                self.conversation_history,
                self.remaining_intents,
            )
            return self._settle_action(payload, response, reaction, elicited, user_ok and coverage_ok)

        feedback, score, judge_response, judge_ok = evaluate_submission(
            payload, self.current_task, self._model_config(), self.conversation_history
        )
        return self._settle_answer(payload, feedback, score, judge_response, judge_ok)

    async def step_async(
        self, action_input: str
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Async version of :meth:`step`."""
        action_str, kind, payload = self._begin_step(action_input)
        if kind == "finish":
            return self._finish_step()
        if kind == "invalid":
            return self._invalid_step(action_str)

        if kind == "action":
            response, reaction, user_ok = await respond_to_agent_async(
                payload, self.current_task, self._model_config(), self.conversation_history
            )
            elicited, coverage_ok = await evaluate_intent_coverage_async(
                payload,
                self.current_task,
                self._model_config(),
                self.conversation_history,
                self.remaining_intents,
            )
            return self._settle_action(payload, response, reaction, elicited, user_ok and coverage_ok)

        feedback, score, judge_response, judge_ok = await evaluate_submission_async(
            payload, self.current_task, self._model_config(), self.conversation_history
        )
        return self._settle_answer(payload, feedback, score, judge_response, judge_ok)

    # ------------------------------------------------------------------
    # Step helpers shared by the sync and async paths
    # ------------------------------------------------------------------

    def _model_config(self) -> Dict[str, Any]:
        return {
            "api_key": self.config.api_key,
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

    def _begin_step(self, action_input: str) -> Tuple[str, str, str]:
        """Advance the step counter and classify the action."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")

        self.step_count += 1
        action_str = str(action_input).strip()
        self.message_history.append(action_str)

        for prefix, kind in (("[action]", "action"), ("[answer]", "answer"), ("[finish]", "finish")):
            if action_str.lower().startswith(prefix):
                return action_str, kind, action_str[8:].strip()
        return action_str, "invalid", ""

    def _observation(self, feedback: str, current_score: float = 0.0, **extra) -> Dict[str, Any]:
        observation = {
            "task_id": self.current_task["id"],
            "goal": GOAL_TEXT,
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "current_score": current_score,
            "best_score": self.best_score,
            "intents_elicited": len(self.elicited_intents),
            "total_intents": len(self.elicited_intents) + len(self.remaining_intents),
        }
        observation.update(extra)
        return observation

    def _info(self, **extra) -> Dict[str, Any]:
        info = {
            "task_id": self.current_task["id"],
            "conversation_history": list(self.conversation_history),
            "message_history": list(self.message_history),
            "reaction_history": list(self.reaction_history),
            "elicited_intents": [dict(i) for i in self.elicited_intents],
            "n_corrections": sum(1 for r in self.reaction_history if r == "correction"),
            "best_score": self.best_score,
            "total_reward": self.total_reward,
        }
        info.update(extra)
        return info

    def _finalize(self, reward: float) -> float:
        """Apply the step penalty and optional normalisation, then bank it."""
        reward -= self.config.step_penalty
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        self.total_reward += reward
        return reward

    def _finish_step(self) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.episode_complete = True
        total = len(self.elicited_intents) + len(self.remaining_intents)
        feedback = (
            f"Session ended. You drew out {len(self.elicited_intents)}/{total} of what the user "
            f"wanted and your best submission scored {self.best_score:.2f}."
        )
        return self._observation(feedback), 0.0, True, False, self._info()

    def _invalid_step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        feedback = "Invalid action format. Please use [action], [answer], or [finish]."
        truncated = self.step_count >= self.config.max_steps
        if truncated:
            self.episode_complete = True
        if self.config.verbose:
            print(f"[SWEGym] Rejected malformed action: {action_str[:80]}")
        return self._observation(feedback), 0.0, False, truncated, self._info(format_ok=False)

    def _intent_reward(self, elicited: List[Dict[str, Any]]) -> float:
        """Importance-weighted credit for the requirements a message surfaced."""
        if not elicited:
            return 0.0
        max_importance = max(int(i.get("importance", 1)) for i in elicited)
        reward = IMPORTANCE_REWARDS.get(max_importance, 0.4)
        # Focused questions are the point; sweeping up many intents at once is
        # discounted the same way IntentionGym discounts multi-part questions.
        if len(elicited) > 1:
            reward = max(0.0, reward - self.config.multi_intent_penalty * (len(elicited) - 1))
        return reward * self.config.reward_scale

    def _settle_action(
        self,
        message: str,
        response: str,
        reaction: str,
        elicited_indices: List[int],
        llm_ok: bool,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        newly_elicited = [self.remaining_intents[i] for i in sorted(elicited_indices)]
        if newly_elicited:
            keep = set(elicited_indices)
            self.elicited_intents.extend(newly_elicited)
            self.remaining_intents = [
                intent for i, intent in enumerate(self.remaining_intents) if i not in keep
            ]

        reward = self._intent_reward(newly_elicited)
        if reaction == "correction":
            reward -= self.config.correction_penalty

        self.conversation_history.append((message, response))
        self.reaction_history.append(reaction)
        reward = self._finalize(reward)

        truncated = self.step_count >= self.config.max_steps
        if truncated:
            self.episode_complete = True

        observation = self._observation(response, current_score=self.best_score, llm_error=not llm_ok)
        info = self._info(
            newly_elicited_intents=[dict(i) for i in newly_elicited],
            reaction=reaction,
            llm_success=llm_ok,
        )
        return observation, reward, False, truncated, info

    def _settle_answer(
        self,
        submission: str,
        feedback: str,
        score: float,
        judge_response: Optional[Dict[str, Any]],
        judge_ok: bool,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # A judge that failed to respond tells us nothing about the submission,
        # so it must not be recorded as a zero-scoring attempt.
        base_reward = 0.0
        if judge_ok:
            base_reward = max(0.0, score - self.best_score)
            if score > self.best_score:
                self.best_score = score
                self.best_judge_response = judge_response
            self.judge_history.append(judge_response)

        self.conversation_history.append((submission, feedback))
        self.reaction_history.append("submission")
        reward = self._finalize(base_reward * self.config.reward_scale)

        terminated = judge_ok and score >= self.config.success_threshold
        truncated = (not terminated) and self.step_count >= self.config.max_steps
        if terminated or truncated:
            self.episode_complete = True

        observation = self._observation(feedback, current_score=score, llm_error=not judge_ok)
        info = self._info(
            submission_score=score if judge_ok else None,
            judge_response=judge_response,
            llm_success=judge_ok,
        )
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def render(self):
        """Print the current session state."""
        total = len(self.elicited_intents) + len(self.remaining_intents)
        print(
            f"[SWEGym] task={self.current_task.get('id', '?')} step={self.step_count} "
            f"intents={len(self.elicited_intents)}/{total} best_score={self.best_score:.2f}"
        )

    def close(self):
        """Nothing to release; the environment holds no external resources."""
        return None
