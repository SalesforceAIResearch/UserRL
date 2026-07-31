"""
Prompts and evaluation functions for SWEGym.

Three LLM roles are used, and they are deliberately kept apart:

* the **simulated user**, who knows what they want (the recorded session's
  script) but not the grading rubric, and who reveals requirements only when
  asked or when the agent drifts;
* the **coverage evaluator**, who knows the oracle intents and decides which of
  them the agent's latest message elicited;
* the **submission judge**, who knows the frozen completeness rubric and scores
  a submitted change against it.

Splitting the user from the evaluator follows IntentionGym: a responder that
could see the reward key would leak it back to the agent.
"""

import json
import re
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI, OpenAI

USER_RESPONSE_SYS = """## **Task**
You are role-playing as the human user in a session with an AI coding agent. You reported a problem and the agent is working on it. Respond to the agent's latest message the way the real user would.

## **Instructions**
1. You have a private session script describing what you want, what you would say, and when you would say it. Follow its spirit rather than quoting it.
2. Do NOT volunteer everything at once. Real users mention a requirement when it becomes relevant, when the agent asks, or when the agent is clearly going the wrong way.
3. If the agent asks a specific question you can answer from the script, answer it directly and concretely.
4. If the agent asks something the script does not cover, answer plausibly and briefly, staying consistent with what you have already said.
5. If the agent is drifting, missing a requirement, or claiming it is done when it is not, push back. Be brief and a little vague, as a real user would be, rather than handing over a checklist.
6. Never mention that a script, rubric, or evaluation exists. Never enumerate your requirements as a numbered list.
7. Classify your own reply with one of these reaction types:
   - "answer": you supplied information the agent asked for
   - "context": you volunteered background the agent had not asked for
   - "new_requirement": you introduced scope the agent did not know about
   - "correction": you told the agent it was wrong or off track
   - "acknowledge": you had nothing substantive to add

## **Example Format**

### Your Response
```json
{
    "thought": "Your reasoning about what the agent said and what the real user would do next.",
    "reaction": "answer" or "context" or "new_requirement" or "correction" or "acknowledge",
    "response": "What you say back to the agent, in the first person."
}
```
"""

USER_RESPONSE_USER = """## **Note**
- Return JSON wrapped in ```json ... ```, parseable, with exactly the fields shown in the example format.
- Keep your response to at most a few sentences. Speak as the user, in the first person.
- Do not reveal the existence of the session script, and do not list your requirements exhaustively.

### The Problem You Reported
<instruction>

### Your Private Session Script
<script>

### Conversation So Far
<history>

### The Agent's Latest Message
<message>

### Your Response
```json
"""

COVERAGE_EVAL_SYS = """## **Task**
You are evaluating how much of a user's underlying intent an AI coding agent surfaced with its latest message. You will be given a list of intents the real user expressed during the original session, and the agent's latest message.

## **Instructions**
1. An intent is "elicited" if the agent's latest message directly asks about it, raises it, or addresses it. Anticipating a requirement before the user states it counts.
2. Only credit intents that the latest message elicited. Do not re-credit intents that earlier messages already covered.
3. Generic pleasantries, status updates, and restatements of the problem elicit nothing. Return an empty list in that case.
4. Be strict. A vague message that could loosely relate to many intents elicits none of them.

## **Example Format**

### Your Response
```json
{
    "analysis": "Brief explanation of which intents, if any, the latest message elicited.",
    "is_task_related": true or false,
    "elicited_intent_indices": [list of indices from the remaining intents list]
}
```
"""

COVERAGE_EVAL_USER = """## **Note**
- Return JSON wrapped in ```json ... ```, parseable, with exactly the fields shown in the example format.
- Indices must refer to the "Remaining Intents" list below, which is numbered from 0.
- You are only evaluating. Do not write a reply to the agent.

### The Problem The User Reported
<instruction>

### Remaining Intents
<intents>

### Conversation So Far
<history>

### The Agent's Latest Message
<message>

### Your Response
```json
"""

JUDGE_SYS = """## **Task**
You are grading an AI coding agent's proposed change against a frozen rubric of completeness goals. You will give a score per goal and short feedback to the agent.

## **Instructions**
1. Score every goal in the rubric, once each, referring to a goal by its index.
2. Each score can only be 0, 0.5, or 1.0. Use 1.0 when the submission clearly achieves the goal, 0.5 when it partially achieves it or describes the right change imprecisely, and 0 when it does not achieve it.
3. Judge what the submission actually says. Do not give credit for intentions, restatements of the problem, or plans the submission does not carry out.
4. In your feedback, say only which parts are on the right track. Do not reveal the rubric, the goals not met, or the intended solution. Be concise and address the agent as "you".

## **Example Format**

### Your Response
```json
{
    "scores": [
        {
            "goal_index": 0,
            "thought": "One short sentence justifying the score.",
            "score": 0 or 0.5 or 1.0
        },
        ... (one entry per rubric goal)
    ],
    "feedback": "Concise feedback about which parts are correct. Reveal nothing about the rubric or the intended solution. Use the second person."
}
```
"""

JUDGE_USER = """## **Note**
- Return JSON wrapped in ```json ... ```, parseable, with exactly the fields shown in the example format.
- The "scores" list must contain exactly one entry per rubric goal, each identified by its "goal_index".
- Keep every "thought" to a single short sentence so the response is not truncated.
- Reason carefully about the submission before scoring. Do not leak the rubric in your feedback.

### The Problem The User Reported
<instruction>

### Conversation So Far
<history>

### The Agent's Submitted Change
<submission>

### Rubric (Completeness Goals)
<rubric>

### Your Response
```json
"""


def _format_history(history: List[Tuple[str, str]], limit: int = 12) -> str:
    """Render the recent agent/user exchanges for a prompt."""
    if not history:
        return "This is the first message in the session."
    lines = []
    for i, (agent_msg, user_msg) in enumerate(history[-limit:], 1):
        lines.append(f"Turn {i}:\nAgent: {agent_msg}\nUser: {user_msg}\n")
    return "\n".join(lines)


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of an LLM response, with or without fences."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    if not candidate.startswith("{"):
        brace = candidate.find("{")
        if brace == -1:
            raise ValueError("No JSON object found in response")
        candidate = candidate[brace:]
    stack, end, in_string, escaped = [], None, False, False
    for i, ch in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                end = i + 1
                break
    if end is not None:
        return json.loads(candidate[:end])
    return _repair_truncated(candidate)


def _repair_truncated(candidate: str) -> Dict[str, Any]:
    """Recover a response the model ran out of tokens mid-way through.

    The judge emits its per-goal scores before its free-text feedback, so a
    truncated response has usually already said everything that affects the
    reward. Close the structures that are still open and keep what parsed.
    """
    for cut in range(len(candidate) - 1, 0, -1):
        if candidate[cut] not in "}]":
            continue
        prefix = candidate[: cut + 1]
        stack, in_string, escaped = [], False, False
        for ch in prefix:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
        if in_string:
            continue
        closing = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        try:
            return json.loads(prefix + closing)
        except json.JSONDecodeError:
            continue
    raise ValueError("Unterminated JSON object in response")


def _messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _complete(model_config: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
    last_error = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                timeout=model_config["timeout"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 - retried, then reported to caller
            last_error = e
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_error}")


async def _complete_async(model_config: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    client = AsyncOpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
    last_error = None
    for _ in range(3):
        try:
            response = await client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                timeout=model_config["timeout"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 - retried, then reported to caller
            last_error = e
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_error}")


# --------------------------------------------------------------------------
# Simulated user
# --------------------------------------------------------------------------

def build_user_prompt(
    message: str, task: Dict[str, Any], history: List[Tuple[str, str]]
) -> List[Dict[str, str]]:
    """Build the prompt for the simulated user's reply."""
    user_prompt = (
        USER_RESPONSE_USER.replace("<instruction>", task["instruction"])
        .replace("<script>", task.get("user_simulation_prompt", "") or "No script available.")
        .replace("<history>", _format_history(history))
        .replace("<message>", message)
    )
    return _messages(USER_RESPONSE_SYS, user_prompt)


def _parse_user_reply(text: str) -> Tuple[str, str]:
    parsed = _extract_json(text)
    response = str(parsed.get("response", "")).strip()
    reaction = str(parsed.get("reaction", "acknowledge")).strip().lower()
    if reaction not in ("answer", "context", "new_requirement", "correction", "acknowledge"):
        reaction = "acknowledge"
    if not response:
        raise ValueError("Simulated user returned an empty response")
    return response, reaction


def respond_to_agent(
    message: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    history: List[Tuple[str, str]],
) -> Tuple[str, str, bool]:
    """
    Generate the simulated user's reply to an agent message.

    Returns:
        Tuple of (response, reaction, success)
    """
    try:
        return (*_parse_user_reply(_complete(model_config, build_user_prompt(message, task, history))), True)
    except Exception as e:  # noqa: BLE001
        print(f"[SWEGym] Error generating user response: {e}")
        return "Sorry, I got distracted. Could you say that again?", "acknowledge", False


async def respond_to_agent_async(
    message: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    history: List[Tuple[str, str]],
) -> Tuple[str, str, bool]:
    """Async version of :func:`respond_to_agent`."""
    try:
        text = await _complete_async(model_config, build_user_prompt(message, task, history))
        return (*_parse_user_reply(text), True)
    except Exception as e:  # noqa: BLE001
        print(f"[SWEGym] Error generating user response: {e}")
        return "Sorry, I got distracted. Could you say that again?", "acknowledge", False


# --------------------------------------------------------------------------
# Intent coverage
# --------------------------------------------------------------------------

def build_coverage_prompt(
    message: str,
    task: Dict[str, Any],
    history: List[Tuple[str, str]],
    remaining_intents: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build the prompt that scores which intents a message elicited."""
    intents_text = "\n".join(
        f"{i}. [{intent.get('kind', 'request')}] {intent['description']}"
        for i, intent in enumerate(remaining_intents)
    )
    user_prompt = (
        COVERAGE_EVAL_USER.replace("<instruction>", task["instruction"])
        .replace("<intents>", intents_text or "No remaining intents.")
        .replace("<history>", _format_history(history))
        .replace("<message>", message)
    )
    return _messages(COVERAGE_EVAL_SYS, user_prompt)


def _parse_coverage(text: str, n_remaining: int) -> List[int]:
    parsed = _extract_json(text)
    indices = parsed.get("elicited_intent_indices", []) or []
    seen, cleaned = set(), []
    for raw in indices:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_remaining and idx not in seen:
            seen.add(idx)
            cleaned.append(idx)
    return cleaned


def evaluate_intent_coverage(
    message: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    history: List[Tuple[str, str]],
    remaining_intents: List[Dict[str, Any]],
) -> Tuple[List[int], bool]:
    """
    Decide which of the remaining intents the agent's message elicited.

    Returns:
        Tuple of (elicited indices into remaining_intents, success)
    """
    if not remaining_intents:
        return [], True
    try:
        text = _complete(
            model_config, build_coverage_prompt(message, task, history, remaining_intents)
        )
        return _parse_coverage(text, len(remaining_intents)), True
    except Exception as e:  # noqa: BLE001
        print(f"[SWEGym] Error evaluating intent coverage: {e}")
        return [], False


async def evaluate_intent_coverage_async(
    message: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    history: List[Tuple[str, str]],
    remaining_intents: List[Dict[str, Any]],
) -> Tuple[List[int], bool]:
    """Async version of :func:`evaluate_intent_coverage`."""
    if not remaining_intents:
        return [], True
    try:
        text = await _complete_async(
            model_config, build_coverage_prompt(message, task, history, remaining_intents)
        )
        return _parse_coverage(text, len(remaining_intents)), True
    except Exception as e:  # noqa: BLE001
        print(f"[SWEGym] Error evaluating intent coverage: {e}")
        return [], False


# --------------------------------------------------------------------------
# Submission judging
# --------------------------------------------------------------------------

def build_judge_prompt(
    submission: str, task: Dict[str, Any], history: List[Tuple[str, str]]
) -> List[Dict[str, str]]:
    """Build the prompt that grades a submitted change against the rubric."""
    rubric = json.dumps(
        [
            {"goal_index": i, "statement": g["statement"]}
            for i, g in enumerate(task["completeness_goals"])
        ],
        indent=4,
    )
    user_prompt = (
        JUDGE_USER.replace("<instruction>", task["instruction"])
        .replace("<history>", _format_history(history))
        .replace("<submission>", submission)
        .replace("<rubric>", rubric)
    )
    return _messages(JUDGE_SYS, user_prompt)


def _parse_judgement(text: str, task: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    parsed = _extract_json(text)
    goals = task["completeness_goals"]
    scores = parsed.get("scores", []) or []

    by_index = {}
    for position, entry in enumerate(scores):
        try:
            # Fall back to positional order if the judge omitted the index.
            index = int(entry.get("goal_index", position))
            raw = float(entry.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(goals):
            # The rubric only defines three levels; snap anything else onto them.
            by_index[index] = min((0.0, 0.5, 1.0), key=lambda allowed: abs(allowed - raw))

    if len(by_index) != len(goals):
        raise ValueError(f"Judge scored {len(by_index)} of {len(goals)} goals")

    total = sum(float(goal["weight"]) * by_index[i] for i, goal in enumerate(goals))

    feedback = str(parsed.get("feedback", "")).strip() or "Reviewed your change."
    return feedback, round(min(1.0, max(0.0, total)), 4), parsed


def evaluate_submission(
    submission: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    history: List[Tuple[str, str]],
) -> Tuple[str, float, Dict[str, Any], bool]:
    """
    Grade a submitted change against the task's frozen rubric.

    Returns:
        Tuple of (feedback, weighted score in [0, 1], raw judge response, success)
    """
    try:
        text = _complete(model_config, build_judge_prompt(submission, task, history))
        return (*_parse_judgement(text, task), True)
    except Exception as e:  # noqa: BLE001
        print(f"[SWEGym] Error judging submission: {e}")
        return "Unable to review your change right now.", 0.0, None, False


async def evaluate_submission_async(
    submission: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    history: List[Tuple[str, str]],
) -> Tuple[str, float, Dict[str, Any], bool]:
    """Async version of :func:`evaluate_submission`."""
    try:
        text = await _complete_async(model_config, build_judge_prompt(submission, task, history))
        return (*_parse_judgement(text, task), True)
    except Exception as e:  # noqa: BLE001
        print(f"[SWEGym] Error judging submission: {e}")
        return "Unable to review your change right now.", 0.0, None, False
