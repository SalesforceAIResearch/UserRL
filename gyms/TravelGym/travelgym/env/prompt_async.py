import json
from typing import Dict, Any, List, Tuple, Optional
import re
import time
import ast
from openai import AsyncOpenAI
import asyncio
import random

from .utils import parse_output_as_json, build_history_into_prompt
from .utils import RESPONSE_PREFERENCE_SYSTEM, RESPONSE_PREFERENCE_USER, RESPONSE_ELICIT_SYSTEM, RESPONSE_ELICIT_USER, RESPONSE_NATURAL_SYSTEM, RESPONSE_NATURAL_USER, JUDGE_PROMPT_SYSTEM, JUDGE_PROMPT_USER, JUDGE_SEARCH_SYSTEM, JUDGE_SEARCH_USER, OPTION_SCHEMAS

async def async_model_call(
    system_prompt: str,
    user_prompt: str,
    model_config: Dict[str, Any],
) -> str:
    """
    Async version of model_call.
    """
    # Create OpenAI async client
    client = AsyncOpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
    # Build the prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try_time = 0
    while try_time < 3:
        try:
            # Make async API call   
            response = await client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                timeout=model_config["timeout"] 
            )
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            response_json = parse_output_as_json(response_text)
            return response_json
        except Exception as e:
            print(f"[TravelGym - Async Model Call] Error calling model: {e}")
            try_time += 1
            if try_time >= 3:
                return None
            await asyncio.sleep(2)
    return None


async def async_generate_judge_search(
    agent_request: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
) -> str:
    """
    Async version of generate_judge_search.
    """
    arguments = task["arguments"]
    arg_string = ""
    for dimension, argument in arguments.items():
        arg_string += f"Aspect Name: {dimension}\n{json.dumps(argument, indent=4)}\n\n"
    arg_string = arg_string.strip()

    system_prompt = JUDGE_SEARCH_SYSTEM
    user_prompt = JUDGE_SEARCH_USER.format(agent_request=agent_request, ground_truth_arguments=arg_string)

    response_json = await async_model_call(system_prompt, user_prompt, model_config)
    return response_json


async def async_generate_judge_response(
    agent_action: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Tuple[str, str]],
    available_preferences: List[Dict[str, Any]]
) -> str:
    """
    Async version of generate_judge_response.
    """
    conversation_history_str = build_history_into_prompt(conversation_history, with_note=True)
    scenario_str = task["scenario"]
    preferences_str = ""
    for preference in available_preferences:
        preferences_str += f"Preference ID: {preference['id']}\tAspect: {preference['aspect']}\nPreference: {preference['preference']}\n\n"
    preferences_str = preferences_str.strip()

    system_prompt = JUDGE_PROMPT_SYSTEM
    user_prompt = JUDGE_PROMPT_USER.format(scenario=scenario_str, conversation_history=conversation_history_str, latest_utterance=agent_action, preferences_list=preferences_str)

    response_json = await async_model_call(system_prompt, user_prompt, model_config)
    return response_json


async def async_evaluate_action(
    action: str,
    task: Dict[str, Any],
    state_config: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Tuple[str, str]],
    available_preferences: List[Dict[str, Any]],
    state_list: Dict[str, Any]
) -> Tuple[str, List[int], float]:
    """
    Async version of evaluate_action.
    
    Args:
        action: The agent's action
        task: Current task dictionary
        state_config: State configuration
        model_config: LLM configuration
        conversation_history: Previous conversation
        available_preferences: Unelicited preferences
        state_list: Current state tracking
        
    Returns:
        Tuple of (response, elicited_preference_indices, reward_info)
    """

    if action.startswith("[search]"):
        action = action[len("[search]"):].strip()
        try:
            state_list["search_times"] += 1
            # Simulate a system error every N times of search
            if state_list["search_times"] % state_config["search_failure_interval"] == 0:
                raise Exception("Simulate a system error")
            
            judgment = await async_generate_judge_search(action, task, model_config)
            assert judgment is not None and "alignment_judgement" in judgment
            if judgment["alignment_judgement"] == "True":
                assert "alignment_aspect" in judgment
                alignment_aspect = judgment["alignment_aspect"].lower().strip()
                alignment_aspect = "rental_car" if (alignment_aspect == "rental car" or alignment_aspect == "car rental") else alignment_aspect
                assert alignment_aspect in task["dimensions"]

                if alignment_aspect in state_list["search_arguments"]:
                    state_list["search_arguments"].remove(alignment_aspect)
                    response_options = task["all_options"][alignment_aspect]
                    schema = OPTION_SCHEMAS[alignment_aspect].strip()
                    feedback = f"You have provided the correct search request arguments.\n\n{schema}\n\nHere are all the options for <{alignment_aspect}>:\n"
                    for option in response_options:
                        feedback += f"{json.dumps(option)}\n"
                    feedback = feedback.strip()
                    conversation_history.append({"role": "agent", "content": action})
                    conversation_history.append({"role": "database", "content": feedback.split("\n")[0] + " ... (skip detailed results here) ..."})
                    return feedback, [], state_config["search_correct_reward"]
                else:
                    feedback = f"You have provided the correct search request arguments. However, you have already got the search results for <{alignment_aspect}> in previous search attempts. Please directly refer to the previous search results."
                    conversation_history.append({"role": "agent", "content": action})
                    conversation_history.append({"role": "database", "content": feedback})
                    return feedback, [], 0.0
            
            elif judgment["alignment_judgement"] == "False":
                feedback = "There's something wrong within your searching request, or the arguments in your request is not accurate. Please refine your search request to 1. align with user's need, 2. make only one request attempt at a time, and 3. cover and only cover correct search request arguments."
                conversation_history.append({"role": "agent", "content": action})
                conversation_history.append({"role": "database", "content": feedback})
                return feedback, [], 0.0

            elif judgment["alignment_judgement"] == "N/A":
                feedback = "You have provided the correct search request arguments. However, the arguments are not related to any of the ground truth search arguments. Please refine your search request to 1. align with user's need, 2. make only one request attempt at a time, and 3. cover and only cover correct search request arguments."
                conversation_history.append({"role": "agent", "content": action})
                conversation_history.append({"role": "database", "content": feedback})
                return feedback, [], 0.0
            
            else:
                raise ValueError(f"Invalid alignment judgement: {judgment['alignment_judgement']}")
            
        except Exception as e:
            if "Simulate a system error" in str(e):
                print(f"[TravelGym - Async Judging Search] Normally simulate a system error")
            else:
                print(f"[TravelGym - Async Judging Search] {e}; By default will return N/A")
            feedback = "Currently the searching backend is experiencing some issues. Please try again later."
            conversation_history.append({"role": "agent", "content": action})
            conversation_history.append({"role": "database", "content": feedback})
            return feedback, [], 0.0

    elif action.startswith("[action]"):
        action = action[len("[action]"):].strip()
        default_type = "1" # 1: normal conversation, 2: preference-related conversation, 3: other conversation
        default_reward = 0.0
        try:
            judgment = await async_generate_judge_response(action, task, model_config, conversation_history, available_preferences)
            assert judgment is not None and "type" in judgment, f"Invalid judgment: {judgment}\nMissing type"
            judgment_type = str(judgment["type"]).strip()
            if judgment_type == "1":
                note = "Agent's utterance is not related to preference or not explicitly asking for a preference. I will respond naturally and coherently."
                reward = 0.0
            elif judgment_type == "2":
                assert "preference_id" in judgment, f"Invalid judgment: {judgment}\nMissing preference_id"
                preference_id = judgment["preference_id"].strip()
                found_preference = False
                current_preference = None
                for preference in available_preferences:
                    if preference["id"] == preference_id:
                        found_preference = True
                        current_preference = preference
                        break
                assert found_preference, f"Invalid judgment: {judgment}\nPreference ID {preference_id} not found in available preferences"
                note = f"Agent's utterance is explicitly asking for a preference that I have. Preference ID: {preference_id}. I will elicit this preference in an implicit and indirect manner."
                reward = state_config["preference_correct_reward"]
            elif judgment_type == "3":
                note = f"Agent's utterance is explicitly asking for a preference, but I do not have this preference. I will respond in a neutral way, but still relevant to the conversation and make sense and coherent."
                reward = 0.0
            elif judgment_type == "4":
                note = f"Agent's utterance is very vague and general, not explicitly asking for a detailed aspect of preference. I will respond by pointing out the agent's utterance is too general and vague, and I will respond naturally and coherently."
                reward = 0.0
            else:
                raise ValueError(f"Invalid agent type: {judgment_type}")
        except Exception as e:
            print(f"[TravelGym - Async Judging Conversation] {e}; By default turn to normal conversation")
            judgment_type = default_type
            note = "Judging of agent's latest utterance failed due to some issues. By default regard it as normal conversation. Should converse naturally and coherently."
            reward = default_reward

        try:
            # elicit the preference that the agent is explicitly asking for
            if judgment_type == "2":
                state_list["nonpreference_times"] = 0
                state_list["active_elicited_preferences"] += 1
                conversation_history_str = build_history_into_prompt(conversation_history, with_note=True)
                preference_str = f"Preference ID: {preference_id}\tAspect: {current_preference['aspect']}\nPreference: {current_preference['preference']}\nImplicit Elicitation Statement: {current_preference['implicit_elicitation']}".strip()
                system_prompt = RESPONSE_PREFERENCE_SYSTEM 
                user_prompt = RESPONSE_PREFERENCE_USER.format(preference=preference_str, conversation_history=conversation_history_str, latest_utterance=action)
                response_json = await async_model_call(system_prompt, user_prompt, model_config)
                # process the model call
                assert response_json is not None and "response" in response_json, f"Invalid response: {response_json}\nMissing response"
                response = response_json["response"].strip()

                conversation_history.append({"role": "agent", "content": action, "note": note})
                conversation_history.append({"role": "user", "content": response})
                return response, [preference_id], reward
            # elicit a preference proactively
            elif state_list["nonpreference_times"] >= state_config["elicitation_interval"] and len(available_preferences) > 0:
                state_list["nonpreference_times"] = 0
                state_list["passive_elicited_preferences"] += 1
                conversation_history_str = build_history_into_prompt(conversation_history, with_note=True)
                # randomly elicit a preference
                random_preference = random.choice(available_preferences)
                preference_id = random_preference["id"]
                preference_str = f"Preference ID: {preference_id}\tAspect: {random_preference['aspect']}\nPreference: {random_preference['preference']}\nImplicit Elicitation Statement: {random_preference['implicit_elicitation']}".strip()
                system_prompt = RESPONSE_ELICIT_SYSTEM
                user_prompt = RESPONSE_ELICIT_USER.format(preference=preference_str, conversation_history=conversation_history_str, latest_utterance=action)
                response_json = await async_model_call(system_prompt, user_prompt, model_config)
                # process the model call
                assert response_json is not None and "response" in response_json, f"Invalid response: {response_json}\nMissing response"
                conversation_history.append({"role": "agent", "content": action, "note": "The agent's latest utterance is not related to any preference I have, and the topic is off the target for several turns. I will respond naturally and coherently, but also proactively elicit a preference in an implicit and indirect manner."})
                conversation_history.append({"role": "user", "content": response})
                return response, [preference_id], reward
            elif judgment_type == "3":
                # the agent's utterance is explicitly asking for a preference, but the specific preference is not available
                state_list["nonpreference_times"] += 1
                response = "This is a good question. However, I do not have specific preference in the aspect you ask about yet (or maybe I have already elicited that to you before). You may continue to ask me about other detailed and specific preferences."
                conversation_history.append({"role": "agent", "content": action, "note": note})
                conversation_history.append({"role": "user", "content": response})
                return response, [], reward
            elif judgment_type == "4":
                # the agent's utterance is too general and vague, not explicitly asking for a detailed
                state_list["nonpreference_times"] += 1
                response = "Your question is too vague and general, and I am not sure how to respond to it. Please ask me about some specific aspects of my preferences, in a more detailed and concrete way, so that I can provide you with a more accurate response."
                conversation_history.append({"role": "agent", "content": action, "note": note})
                conversation_history.append({"role": "user", "content": response})
                return response, [], reward
            else:
                # fallback to default response through natural conversation
                state_list["nonpreference_times"] += 1
                conversation_history_str = build_history_into_prompt(conversation_history, with_note=True)
                system_prompt = RESPONSE_NATURAL_SYSTEM
                user_prompt = RESPONSE_NATURAL_USER.format(conversation_history=conversation_history_str, latest_utterance=action)
                response_json = await async_model_call(system_prompt, user_prompt, model_config)
                # process the model call
                assert response_json is not None and "response" in response_json, f"Invalid response: {response_json}\nMissing response"
                response = response_json["response"].strip()
                conversation_history.append({"role": "agent", "content": action, "note": note})
                conversation_history.append({"role": "user", "content": response})
                return response, [], reward
        
        except Exception as e:
            print(f"[TravelGym - Async Responding to Agent] {e}; Error in responding to agent's latest utterance")
            note = "Responding system met some issues. Fallback to default pre-defined response."
            reward = default_reward
            conversation_history.append({"role": "agent", "content": action, "note": note})
            conversation_history.append({"role": "user", "content": "I'm sorry, I'm not sure how to respond to your latest utterance right now. Please try again."})
            return "I'm sorry, I'm not sure how to respond to your latest utterance right now. Please try again.", [], reward

    elif action.startswith("[answer]"):
        action = action[len("[answer]"):].strip()
        answer_ids = action.split(",")
        answer_ids = [id.strip() for id in answer_ids]
        reward = 0.0
        best_found = False
        correct_found = False
        wrong_found = False
        for id in answer_ids:
            # Validate the ID format: an uppercase letter from "ACFHR" followed by digits
            if not re.match(r"^[ACFHR]\d+$", id):
                feedback = f"Invalid option ID format detected for '{id}'. Expected format: H1, F12, C4, etc."
                return feedback, [], 0.0

            id_initial = id[0]
            initial_aspect_map = {"F": "flight", "A": "apartment", "C": "rental_car", "H": "hotel", "R": "restaurant"}

            if state_config["one_choice_per_aspect"] and id_initial in state_list["choice_initials"]:
                feedback = f"You have already recommended an option with the same initial '{id_initial}'. You are allowed to recommend only one option per traveling aspect. Please focus on the search, user preference, and option recommendation on other traveling aspects now."
                return feedback, [], 0.0

            if id in state_list["remaining_best_options"]:
                reward += state_config["choice_best_reward"]
                best_found = True
                state_list["remaining_best_options"].remove(id)
                state_list["remaining_correct_options"].remove(id)
            elif id in state_list["remaining_correct_options"]:
                reward += state_config["choice_correct_reward"]
                correct_found = True
                state_list["remaining_correct_options"].remove(id)
            else:
                reward -= state_config["wrong_choice_penalty"]
                wrong_found = True
            
            if id_initial not in state_list["choice_initials"]:
                state_list["choice_initials"].append(id_initial)

            # remove the remaining preferences of certain aspect if the option is chosen
            if state_config["one_choice_per_aspect"] and id_initial in initial_aspect_map:
                pref_remove_idx = []
                for i, pref in enumerate(available_preferences):
                    # remove the preference if it matches the initial aspect of the chosen option
                    if pref["aspect"] == initial_aspect_map[id_initial]:
                        pref_remove_idx.append(i)
                # Remove preferences in reverse order to avoid index shifting
                for i in reversed(pref_remove_idx):
                    available_preferences.pop(i)

        reward = max(reward, 0.0)

        if best_found and not wrong_found:
            feedback = f"Your chosen options contain the best option!"
        elif best_found and wrong_found:
            feedback = f"Your chosen options contain the best option, but also contain some wrong options."
        elif correct_found and not wrong_found:
            feedback = f"Your chosen options contain the correct option, but not the best option."
        elif correct_found and wrong_found:
            feedback = f"Your chosen options contain the correct option, but not the best option, and also contain some wrong options."
        else:
            feedback = f"Your chosen options do not contain any of the best or correct options."
        
        if state_config["one_choice_per_aspect"]:
            feedback = feedback.strip() + f" Your choice is recorded and do not choose options of this travel aspect again. Please continue your interaction and reasoning focusing on other travel aspects."
        else:
            feedback = feedback.strip() + f" Your choice is recorded and do not choose these options again. Please continue your interaction and reasoning."

        return feedback, [], reward
    
    else:
        return "Your response format is wrong and cannot be parsed properly.", [], 0.0