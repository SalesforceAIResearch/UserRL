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
Preprocess the SearchGym dataset to parquet format
"""

import argparse
import os
import re
import json
from datasets import Dataset
import numpy as np

from verl.utils.hdfs_io import copy, makedirs

np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/fsx/home/cqian/projects/dataset/alfworld_multiturn")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/fsx/home/cqian/projects/AlfworldGym/alfworldgym/data/eval_in_distribution.json"
    dataset = json.load(open(data_source))
    dataset_len = len(dataset)

    filtered_dataset = []
    for item in dataset:
        id = int(item["id"])
        admissible_commands = item["info"]["admissible_commands"][0]
        goto_count = 0
        command_string = "In your next action, the content could be following: "
        for cmd in admissible_commands:
            if cmd.startswith("go to"):
                goto_count += 1
                if goto_count == 1:
                    command_string += cmd.strip() + ", "
                elif goto_count == 2:
                    command_string += cmd.split("go to")[1].strip() + ", "
                elif goto_count == 3:
                    command_string += "etc. (any receptacles presented); "
            else:
                command_string += cmd.strip() + "; "
        command_string = command_string.strip().rstrip(";") + ". According to your task goal, please wisely choose your next action through calling the tool."
        obs = item["obs"].split("ALFRED! =-")[1].strip()

        user_command = obs + "\n\n" + command_string
        filtered_dataset.append({
            "id": id,
            "user_command": user_command,
        })

        print(user_command)
    
    dataset = filtered_dataset
    dataset_len = len(dataset)
    
    # let the value to form a list
    np.random.shuffle(dataset)
    
    # train_dataset = dataset[:int(dataset_len * 0.9)]
    test_dataset = dataset
    
    print(f"test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        id = str(example.pop("id"))
        user_command = example.pop("user_command")

        data = {
            "data_source": "interact_alfworldgym",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                            "You are an agent that actively interact with an specific environment. The followings are the details of the environment and your action space.\n\n"
                            "- Environment Description: AlfworldGym is a text based home environment where you should follow the command in accomplishing the household tasks. You should thoroughly understand your goal, make plan of how to achieve your goal, and perform action step by step.\n\n"
                            "- Action Space: You should invocate the function `interact_with_env` to interact with the environment. The action you can perform is constrained to `action`, and for the action content you shall follow certain format.\n\n"
                            "- Action Description:\n"
                            "  * `action`: The action you would like to take in the environment in your next step. Note the the content of your action should be one of the following\n"
                            "    - look: look around your current location\n"
                            "    - inventory: check your current inventory\n"
                            "    - go to (receptacle): move to a receptacle\n"
                            "    - open (receptacle): open a receptacle\n"
                            "    - close (receptacle): close a receptacle\n"
                            "    - take (object) from (receptacle): take an object from a receptacle\n"
                            "    - move (object) to (receptacle): place an object in or on a receptacle\n"
                            "    - examine (something): examine a receptacle or an object\n"
                            "    - use (object): use an object\n"
                            "    - heat (object) with (receptacle): heat an object using a receptacle\n"
                            "    - clean (object) with (receptacle): clean an object using a receptacle\n"
                            "    - cool (object) with (receptacle): cool an object using a receptacle\n"
                            "    - slice (object) with (object): slice an object using a sharp object\n"
                            "    Please also note that admissble action contents are also provided to you in the environment observation each turn. Please also strategically refer to this information when giving your action content.\n\n"
                            "- Important Notes:\n"
                            "  * In each round of your interaction, you should analyze and carefully consider what to do next, and then invocate the `interact_with_env` tool to interact with the environment. You should provide your thought in each step when invocate the tool.\n"
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly make plan about how to accomplish the task based on your understanding of the environment and the available actions. You may need to strategically make executable plans and try diverse ways to reach the task goal.\n"
                            "  * Usually you should first make a plan about how to reach the task goal, and during each turn you should carefully analysis the current situation, adjust your plan accordingly, and then give your next action following the instructed format.\n"
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
                        ),
                },
                {
                    "role": "user",
                    "content": user_command,
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": "", "env_name": "AlfworldGym", "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "AlfworldGym", "id": id},
                    },
                },
            },
        }
        return data


    # train_dataset = [make_map_fn(example, idx, "train") for idx, example in enumerate(train_dataset)]
    test_dataset = [make_map_fn(example, idx, "test") for idx, example in enumerate(test_dataset)]
    
    # Make it into Dataset with features
    # train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    os.makedirs(local_dir, exist_ok=True)
    
    # train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
