# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the TurtleGym dataset to parquet format
"""

import argparse
import os
import json
from datasets import Dataset
import numpy as np

np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/intention_multiturn")

    args = parser.parse_args()

    train_data_source = "./gyms/IntentionGym/intentiongym/data/all_intentions_refined.json"
    test_data_source = "./gyms/IntentionGym/intentiongym/data/all_intentions_test.json"
    train_dataset = json.load(open(train_data_source))
    test_dataset = json.load(open(test_data_source))
    
    # let the value to form a list
    np.random.shuffle(train_dataset)
    train_dataset = train_dataset[:380]
    np.random.shuffle(test_dataset)
    test_dataset = test_dataset[:40]

    train_dataset_len = len(train_dataset)
    test_dataset_len = len(test_dataset)
    
    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        task = example.pop("task")
        category = example.pop("category")
        missing_details = example.pop("missing_details")
        id = example.pop("id")

        missing_desc = ", ".join([detail["description"] for detail in missing_details])

        data = {
            "data_source": "interact_intention",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n"
                            "- Environment Description: IntentionGym is a scenario where you're given a vague agent task. Your goal is to clarify the missing details in the task and try to get them through queries in conversation.\n\n"
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `action`.\n\n"
                            "- Action Description:\n"
                            "  * `action`: If you choose `action`, you must provide a clarifying question in the field `content` to interact with the environment. The aim of your provided question should be to clarify the missing details in initially given task.\n\n"
                            "- Important Notes:\n"
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n"
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly choose your clarification strategy, so that you can figure out all the missing details about the task in the most efficient way.\n"
                            "  * Usually you should try to figure out all the possible missing details about the given initial task, and provide only one clarifying question in each turn. You could ask the clarifying question for multiple turns by continuing interacting with the environment.\n"
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
                        ),
                },
                {
                    "role": "user",
                    "content": (
                            f"The category of the task: {category}\n"
                            f"The initial vague task description: {task}\n"
                            "Try to figure out all the missing details about this through asking clarification questions!\n"
                    ),
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": missing_desc, "env_name": "IntentionGym", "task": task, "category": category, "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "IntentionGym", "task": task, "category": category, "id": id},
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
