"""
Preprocess the TauGym dataset to parquet format
"""

import argparse
import os
import re
import json
from datasets import Dataset
import numpy as np

np.random.seed(42)

def preprocess_tau_dataset(dataset):
    new_dataset = []
    for item in dataset:
        id = item["id"]
        user_command = item["obs"].strip()
        new_dataset.append({
            "id": id,
            "user_command": user_command,
        })
    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/tau_multiturn")

    args = parser.parse_args()

    data_source = "./gyms/TauGym/taugym/data/tau_train.json"
    train_dataset = json.load(open(data_source))
    test_data_source = "./gyms/TauGym/taugym/data/tau_test.json"
    test_dataset = json.load(open(test_data_source))

    train_dataset = preprocess_tau_dataset(train_dataset)
    test_dataset = preprocess_tau_dataset(test_dataset)

    print(f"train_dataset: {len(train_dataset)}")
    print(f"test_dataset: {len(test_dataset)}")

    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        id = example.pop("id")
        user_command = example.pop("user_command")

        data = {
            "data_source": "interact_taugym",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n"
                            "- Environment Description: TauGym is an environment where you need to interact with both the user and internal tools to fulfill the user's request. You should thoroughly understand the user's goal, figure out what information is needed, and get these information through querying the user or leveraging the internal tool step by step.\n\n"
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `search`, `action` or `answer`.\n\n"
                            "- Action Description:\n"
                            "  * `search`: If you choose `search`, you must specify either 'tools' or 'help' in the `content` field. Giving 'tools' will return a list of internal tools, including their descriptions and required arguments, which you can later call through choosing `answer`. Giving 'help' will return a general guidance on how to interact with the environment effectively.\n"
                            "  * `action`: If you choose `action`, you will communicate directly with the user through the message you write in the `content` field. Ask clear and specific questions to gather the information needed to fulfill the user's request. Keep in mind that the user may not have all the necessary details, so you might need to both request additional user input and call internal tools step by step to reach the goal.\n"
                            "  * `answer`: If you choose `answer`, you must provide an internal tool call in the `content` field, with the tool name and its arguments in JSON format (e.g. {\"name\": tool_name, \"arguments\": {\"arg_1\": \"value_1\", \"arg_2\": \"value_2\"}})\n\n"
                            "- Important Notes:\n"
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n"
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly balance the number of rounds you call the internal tool or communicate with the user, so that you can fulfill the user's request in the most efficient way.\n"
                            "  * Usually you should first search for all available internal tools, understand the user's request and how to fulfill it by calling the internal tool. Then you should analyze the information (argument contents) you need, and how to get these information either through calling other internal tools or directly asking the user. Keep track of what information you already have and what you need, and adjust your plan accordingly.\n"
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
                        ),
                },
                {
                    "role": "user",
                    "content": (
                            f"Here is my request: {user_command}\n"
                            "Try to fulfill it step by step by calling the internal tools and communicating with me!\n"
                    ),
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": "", "env_name": "TauGym", "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "TauGym", "id": id},
                    },
                },
            },
        }
        return data


    train_dataset = [make_map_fn(example, idx, "train") for idx, example in enumerate(train_dataset)]
    test_dataset = [make_map_fn(example, idx, "test") for idx, example in enumerate(test_dataset)]
    
    # Make it into Dataset with features
    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)

    local_dir = args.local_dir
    
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
