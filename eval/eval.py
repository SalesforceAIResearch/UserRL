import os
import json
import time
import argparse
import yaml
import copy
from openai import AsyncOpenAI
from google.genai import Client, types
import pandas as pd
import asyncio
import hashlib

MAX_WORKER_NUM = int(os.environ.get("MAX_WORKER_NUM", 25))
semaphore = asyncio.Semaphore(MAX_WORKER_NUM)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "./") # Please change the path to the `eval` folder
    
async def load_data(env_name, one_choice=True, split="test"):
    if one_choice and "travel" in env_name:
        path = f"{PROJECT_ROOT}/../data/{env_name}_multiturn_onechoice/{split}.parquet"
    else:
        path = f"{PROJECT_ROOT}/../data/{env_name}_multiturn/{split}.parquet"
    df = pd.read_parquet(path)
     # turn into a list of data, using only the prompt
    data = []
    for i in range(len(df)):
        data.append({
            "env_name": env_name,
            "gold": str(df.iloc[i]["reward_model"]["id"]) if (env_name == "intention" or env_name == "persuasion" or env_name == "bamboogle" or env_name == "alfworld" or env_name == "tau" or env_name == "function" or "travel" in env_name) else str(df.iloc[i]["reward_model"]["title"]),
            "messages": list(df.iloc[i]["prompt"]),
        })
    print(f"Loaded {len(data)} data from {path}")
    return data


async def build_env(data, max_turns):
    gold = data["gold"]
    env_name = data["env_name"]
    model_name = os.environ.get("USER_MODEL_NAME", "gpt-4o")

    print("Building environment...", env_name)
    if "travel" in env_name:
        import travelgym
        config = travelgym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        # configure the one choice option
        config.one_choice_per_aspect = True
        config.search_correct_reward = 0.2
        config.preference_correct_reward = 0.6
        env = travelgym.TravelEnv(config=config)
        env.reset()
        return env
    elif env_name == "turtle":
        import turtlegym
        config = turtlegym.get_default_config()
        config.max_steps = max_turns
        config.success_threshold = 1.0
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        env = turtlegym.StoryEnv(config=config)
        env.reset()
        return env
    elif env_name == "telepathy":
        import telepathygym
        config = telepathygym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        env = telepathygym.TelepathyEnv(config=config)
        env.reset()
        return env
    elif env_name == "intention":
        import intentiongym
        config = intentiongym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        env = intentiongym.IntentionEnv(config=config)
        env.reset()
        return env
    elif env_name == "persuasion":
        import persuadegym
        config = persuadegym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        env = persuadegym.PersuadeEnv(config=config)
        env.reset()
        return env
    elif env_name == "bamboogle":
        import searchgym
        config = searchgym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.eval_method = "llm"
        config.model_name = model_name
        # 5 is default value
        # config.max_search_results = 5
        # config.max_search_steps = 5
        env = searchgym.SearchEnv(config=config)
        env.reset()
        return env
    elif env_name == "alfworld":
        import alfworldgym
        config = alfworldgym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = int(gold)
        env = alfworldgym.AlfworldEnv(config=config)
        env.reset()
        return env
    elif env_name == "tau":
        import taugym
        config = taugym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.user_model = model_name
        # special handling for task_category and task_split
        if "retail" in gold:
            config.task_category = "retail"
        else:
            config.task_category = "airline"
        if "train" in gold:
            config.task_split = "train"
        else:
            config.task_split = "test"
        env = taugym.TauEnv(config=config)
        env.reset()
        return env
    elif env_name == "function":
        import functiongym
        config = functiongym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        env = functiongym.FunctionEnv(config=config)
        env.reset()
        return env
    else:
        raise ValueError(f"Environment {env_name} not supported")


async def gen_response(client, data, schema, temperature, model_name):
    for _ in range(10):
        try:
            if "gemini" in model_name:
                interact_tool = copy.deepcopy(schema["function"])
                tools = types.Tool(function_declarations=[interact_tool])
                config = types.GenerateContentConfig(
                    tools=[tools],
                    temperature=temperature,
                    max_output_tokens=2048,
                    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY))
                )
                contents = [
                    types.Content(role="user", parts=[
                        types.Part(text=data["messages"][0]["content"]),
                        types.Part(text=data["messages"][1]["content"])
                    ])
                ]
                for message in data["messages"][2:]:
                    contents.append(message["content"])
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                part_exists = response.candidates[0].content.parts[0]
                return response
            
            else:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=data["messages"],
                    tools=[schema],
                    tool_choice="required" if str(os.environ.get("TOOL_CHOICE", "required")) == "required" else "auto",
                    temperature=temperature,
                    max_tokens=2048,
                    n=1,
                    # reasoning_effort="high"
                )
                return response
        
        except Exception as e:
            print(f"[local tool call] failed: {e}")
            time.sleep(2)
    
    print(f"[local tool call] failed three times, please check the model name and API key.")
    raise RuntimeError("!!! Local function_call failed three times !!!")


def hash(data):
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


async def rollout(data, client, function, temperature, max_turns, model_name):
    turn = 0
    rewards = []
    hash_id = hash(data)
    history = []
    json_data = copy.deepcopy(data)

    try:
        env = await build_env(data, max_turns)
        while turn < max_turns:
            if "gemini" in model_name:
                response = await gen_response(client, data, function, temperature, model_name)
                # print(f"Model response: {response}")
                try:
                    json_args = response.candidates[0].content.parts[0].function_call.args
                    if isinstance(json_args, str):
                        json_args = json.loads(json_args)
                except Exception as e:
                    assert False, f"Invalid response (response without tool call detected): {response}"
            else:
                response = await gen_response(client, data, function, temperature, model_name)
                print(f"Model response: {response}")
                try:
                    args = response.choices[0].message.tool_calls[0].function.arguments
                    if isinstance(args, str):
                        json_args = json.loads(args)
                    else:
                        json_args = args
                except Exception as e:
                    assert False, f"Invalid response (response without tool call detected): {response}"

            assert "choice" in json_args and "content" in json_args, f"Invalid response: {args}"

            choice = json_args["choice"]
            content = json_args["content"]
            if choice == "action" and not content.startswith("[action]"):
                formatted_action = "[action] " + content
            elif choice == "answer" and not content.startswith("[answer]"):
                formatted_action = "[answer] " + content
            elif choice == "search" and not content.startswith("[search]"):
                formatted_action = "[search] " + content
            else:
                formatted_action = content
            
            observation, reward, terminated, truncated, info = await asyncio.wait_for(
                env.step_async(formatted_action),
                timeout=120.0
            )
            feedback = observation["feedback"]

            if len(feedback) > 512 and choice == "search":
                output_feedback = feedback[:256] + "  ... ... " + feedback[-256:]
            else:
                output_feedback = feedback
            
            print(f"In {data['env_name']}, turn {turn}, action: {formatted_action}, feedback: {output_feedback}, reward: {reward}")
            rewards.append(reward)

            history.append({
                "turn": turn,
                "choice": choice,
                "content": content,
                "feedback": feedback,
                "reward": reward,
            })

            json_data["messages"].append({
                "role": "assistant",
                "content": {"name": "interact_with_env", "arguments": json_args}
            })
            json_data["messages"].append({
                "role": "tool",
                "content": feedback
            })

            if terminated or truncated:
                break

            if "gpt" in model_name:
                tool_call = response.choices[0].message.tool_calls[0]
                content = response.choices[0].message.content
                data["messages"].append({"role": "assistant", "tool_calls": [tool_call], "content": content})
                data["messages"].append({"role": "tool", "tool_call_id": tool_call.id, "content": feedback})
            elif "gemini" in model_name:
                tool_call = response.candidates[0].content.parts[0].function_call
                function_response_part = types.Part.from_function_response(name=tool_call.name, response={"result": feedback})
                response_content = types.Content(role="user", parts=[function_response_part])
                data["messages"].append({"role": "assistant", "content": response.candidates[0].content})
                data["messages"].append({"role": "tool", "content": response_content})
            else:
                tool_call = response.choices[0].message.tool_calls[0]
                try:
                    content = response.choices[0].message.content
                    content = "" if not content else content
                except:
                    content = ""
                # Add this for reasoning models like qwen3
                try:
                    reasoning_content = response.choices[0].message.reasoning_content
                    reasoning_content = "" if not reasoning_content else "<think>" + reasoning_content + "</think>"
                except:
                    reasoning_content = ""
                final_content = reasoning_content + content
                data["messages"].append({"role": "assistant", "tool_calls": [tool_call], "content": final_content})
                data["messages"].append({"role": "tool", "content": feedback})
            turn += 1
            
        total_reward = rewards
        return {"hash_id": hash_id, "reward": total_reward, "history": history, "data": json_data}
    
    except asyncio.TimeoutError:
        print(f"==================== [local] rollout timeout !!! ====================")
        total_reward = rewards if len(rewards) > 0 else [0]
        return {"hash_id": hash_id, "reward": total_reward, "history": history, "data": json_data}
    
    except Exception as e:
        print(f"==================== [local] rollout failed: {e} ====================")
        total_reward = rewards if len(rewards) > 0 else [0]
        return {"hash_id": hash_id, "reward": total_reward, "history": history, "data": json_data}


async def post_process_results(results, reward_cache, env, pass_k):
    if "travel" in env:
        results[env][str(pass_k)] = {}
        number_of_1 = []
        number_of_08 = []
        micro_avg = []
        micro_max = []
        for hash_id in reward_cache[env]:
            turn_scores = reward_cache[env][hash_id]["reward"][-1]
            number_of_1.append(turn_scores.count(1.0)) # best choice
            number_of_08.append(turn_scores.count(0.8)) # correct choice
            micro_avg.append(sum(turn_scores) / len(turn_scores) if len(turn_scores) > 0 else 0)
            micro_max.append(max(turn_scores) if len(turn_scores) > 0 else 0)
        results[env][str(pass_k)]["micro_avg"] = sum(micro_avg) / len(micro_avg)
        results[env][str(pass_k)]["micro_max"] = sum(micro_max) / len(micro_max)
        results[env][str(pass_k)]["avg_number_of_08"] = sum(number_of_08) / len(number_of_08)
        results[env][str(pass_k)]["avg_number_of_1"] = sum(number_of_1) / len(number_of_1)
        print(f"\n ######### Pass {pass_k} reward: {results[env][str(pass_k)]["micro_max"]} ######### \n")
    else: # env in ["turtle", "telepathy", "intention", "persuasion", etc.]
        results[env][str(pass_k)] = {}
        micro_avg = []
        for hash_id in reward_cache[env]:
            all_rewards = reward_cache[env][hash_id]["reward"]
            all_rewards = [sum(r) if isinstance(r, list) else r for r in all_rewards]
            micro_avg.append(sum(all_rewards) / len(all_rewards) if len(all_rewards) > 0 else 0)
        results[env][str(pass_k)]["micro_avg"] = sum(micro_avg) / len(micro_avg)
        micro_max = []
        for hash_id in reward_cache[env]:
            all_rewards = reward_cache[env][hash_id]["reward"]
            all_rewards = [sum(r) if isinstance(r, list) else r for r in all_rewards]
            micro_max.append(max(all_rewards) if len(all_rewards) > 0 else 0)
        results[env][str(pass_k)]["micro_max"] = sum(micro_max) / len(micro_max)
        print(f"\n ######### Pass {pass_k} reward: {results[env][str(pass_k)]["micro_max"]} ######### \n")
    return results


async def limited_rollout(*args, **kwargs):
    async with semaphore:
        return await rollout(*args, **kwargs)
    

async def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    arg_parser.add_argument("--port", type=int, default=8000)
    arg_parser.add_argument("--max_turns", type=int, default=8)
    arg_parser.add_argument("--pass_k", type=int, nargs="+", default=[1])
    arg_parser.add_argument("--temperature", type=float, default=1.0)
    arg_parser.add_argument("--envs", type=str, nargs="+", default=["travel22", "travel33", "travel44"])
    arg_parser.add_argument("--save_name", type=str, default="results")

    args = arg_parser.parse_args()

    print(args)

    if "gpt" in args.model_name and "gpt-oss" not in args.model_name:
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")
    elif "gemini" in args.model_name:
        client = Client(api_key=os.environ["GENAI_API_KEY"])
    else:
        base_url = f"http://localhost:{args.port}/v1"
        client = AsyncOpenAI(api_key="dummy", base_url=base_url)

    function = yaml.safe_load(open(f"{PROJECT_ROOT}/schema/interact_tool.yaml", "r"))["tool_schema"]

    if os.path.exists(f"{PROJECT_ROOT}/{args.save_name}_results.json"):
        results = json.load(open(f"{PROJECT_ROOT}/{args.save_name}_results.json", "r"))
    else:
        results = {}
    if os.path.exists(f"{PROJECT_ROOT}/{args.save_name}_reward_cache.json"):
        reward_cache = json.load(open(f"{PROJECT_ROOT}/{args.save_name}_reward_cache.json", "r"))
    else:
        reward_cache = {}
    
    for env in args.envs:
        print(f"Evaluating {env}...")
        data = await load_data(env)
        if env not in results:
            results[env] = {}
        if env not in reward_cache:
            reward_cache[env] = {}
        
        for pass_k in args.pass_k:
            if str(pass_k) in results[env]:
                print(f"Pass {pass_k} already evaluated, skipping...")
                continue

            reqs = []
            # use limited rollout to avoid getting stuck
            for d in data:
                data_id = hash(d)
                existing_number = len(reward_cache[env][data_id]["reward"]) if data_id in reward_cache[env] else 0
                needed_number = max(0, pass_k - existing_number)
                if needed_number == 0:
                    print(f"Data {data_id} already has enough rollouts, skipping...")
                    continue
                for _ in range(needed_number):
                    reqs.append(
                        limited_rollout(copy.deepcopy(d), client, function, args.temperature, args.max_turns, args.model_name)
                    )

            # Run all rollout requests in parallel
            rewards = await asyncio.gather(*reqs)

            # Process the rewards
            for r in rewards:
                hash_id = r["hash_id"]
                reward = r["reward"]
                history = r["history"]
                data = r["data"]
                post_processed_reward = reward # if "travel" in env else sum(reward)
                if hash_id not in reward_cache[env]:
                    reward_cache[env][hash_id] = {"history": [], "reward": [], "data": []}
                reward_cache[env][hash_id]["history"].append(history)
                reward_cache[env][hash_id]["data"].append(data)
                reward_cache[env][hash_id]["reward"].append(post_processed_reward)
            
            # Post-process the results
            results = await post_process_results(results, reward_cache, env, pass_k)

            save_dir = args.save_name.rsplit("/", 1)[0]
            if not os.path.exists(f"{PROJECT_ROOT}/{save_dir}"):
                os.makedirs(f"{PROJECT_ROOT}/{save_dir}")

            with open(f"{PROJECT_ROOT}/{args.save_name}_results.json", "w") as f:
                json.dump(results, f, indent=4)
            with open(f"{PROJECT_ROOT}/{args.save_name}_reward_cache.json", "w") as f:
                json.dump(reward_cache, f, indent=4)
        

if __name__ == "__main__":
    asyncio.run(main())


    
    
    