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

from verl.utils.hdfs_io import copy, makedirs

np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/fsx/home/cqian/projects/dataset/persuasion_multiturn")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/fsx/home/cqian/projects/PersuadeGym/persuadegym/data/all_statements_refined.json"
    dataset = json.load(open(data_source))
    
    # let the value to form a list
    np.random.shuffle(dataset)
    dataset = dataset[:420]
    dataset_len = len(dataset)

    train_dataset = dataset[:int(dataset_len * 0.9)]
    test_dataset = dataset[int(dataset_len * 0.9):]
    
    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        claim = example.pop("claim")
        argument = example.pop("argument")
        id = example.pop("id")

        data = {
            "data_source": "interact_persuade",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n"
                            "- Environment Description: PersuadeGym is a persuasion game where you're given a statement that the environment holds, but you don't agree with it. Your goal is to persuade the environment to change its position on the statement.\n\n"
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `answer`.\n\n"
                            "- Action Description:\n"
                            "  * `answer`: If you choose `answer`, you must provide an argument in the `content` field to interact with the environment. The argument should be a response to express why you oppose the statement. You should respond to the environment's position and try to persuade it to change its original position.\n\n"
                            "- Important Notes:\n"
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n"
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly choose the strategy, so that you can persuade the environment in the most efficient way.\n"
                            "  * Usually you should first analyze any weaknesses or loopholes in the environment's position or arguments. Then, respond with your own arguments and claims in a logical, coherent, and persuasive manner.\n"
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
                            ),
                    },
                {
                    "role": "user",
                    "content": (
                            f"Claim that you oppose but the environment holds: {claim}\n"
                            "Try to persuade the environment to change its position regarding this claim!\n"
                    ),
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": claim, "env_name": "PersuadeGym", "claim": claim, "argument": argument, "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "PersuadeGym", "claim": claim, "argument": argument, "id": id},
                        # "execute_kwargs": {},
                        # "calc_reward_kwargs": {},
                        # "release_kwargs": {},
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
    hdfs_dir = args.hdfs_dir
    
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
