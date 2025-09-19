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
import re
import json
from datasets import Dataset
import numpy as np

from verl.utils.hdfs_io import copy, makedirs

np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/fsx/home/cqian/projects/dataset/turtle_multiturn")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/fsx/home/cqian/projects/TurtleGym/turtlegym/data/all_stories_refined.json"
    dataset = json.load(open(data_source))
    dataset_len = len(dataset)
    
    # let the value to form a list
    dataset = list(dataset.values())
    np.random.shuffle(dataset)
    
    train_dataset = dataset[:int(dataset_len * 0.9)]
    test_dataset = dataset[int(dataset_len * 0.9):]
    
    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        surface = example.pop("new_surface")
        bottom = example.pop("new_bottom")
        title = example.pop("new_title")
        protocol = example.pop("evaluation")

        data = {
            "data_source": "interact_turtlegym",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n"
                            "- Environment Description: TurtleGym is a storytelling game where you're given a brief story snippet (the 'surface') that hints at a deeper, hidden narrative (the 'bottom'). Your goal is to uncover or imagine the full context behind the surface, revealing the twist or true meaning of the story.\n\n"
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `action` or `answer`.\n\n"
                            "- Action Description:\n"
                            "  * `action`: If you choose `action`, you must provide a Yes-or-No question in the `content` field to interact with the environment. Use this action to gather more information about the hidden narrative and delve deeper into the story.\n"
                            "  * `answer`: If you choose `answer`, you must provide a version of the story in the `content` field that incorporates all inferred hidden twists and represents the most plausible interpretation based on the given surface details, your proposed questions, and the corresponding environment responses.\n\n"
                            "- Important Notes:\n"
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n"
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly balance the number of rounds that you take action or provide answer, so that you can uncover the hidden narrative in the most efficient way.\n"
                            "  * Usually you should first take action to validate your wild guesses by asking questions and uncovering the hidden truth. Then, provide your answer story, ensuring it comprehensively reflects all the details you have guessed or uncovered. You can provide your answer at any time and multiple times.\n"
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
                        ),
                },
                {
                    "role": "user",
                    "content": (
                            f"Title: {title}\nSurface: {surface}\n"
                            "Try to uncover the hidden narrative behind the surface, revealing the twist or true meaning of the story!\n"
                    ),
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": bottom, "env_name": "TurtleGym", "surface": surface, "bottom": bottom, "title": title},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "TurtleGym", "surface": surface, "bottom": bottom, "title": title},
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
