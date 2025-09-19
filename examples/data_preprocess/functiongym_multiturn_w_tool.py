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
Preprocess the FunctionGym dataset to parquet format
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
    parser.add_argument("--local_dir", default="/fsx/home/cqian/projects/dataset/function_multiturn")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/fsx/home/cqian/projects/FunctionGym/functiongym/data/functions.json"
    dataset = json.load(open(data_source))
    
    # let the value to form a list
    np.random.shuffle(dataset)
    dataset_len = len(dataset)

    train_dataset = dataset[:460]
    test_dataset = dataset[460:]
    
    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        id = example.pop("id")
        expected_result = str(example.pop("expected_result"))

        data = {
            "data_source": "interact_function",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n"
                            "- Environment Description: FunctionGym is a mapping function guessing game. You input four numbers and receive an output based on a hidden function involving only simple operations: addition, subtraction, multiplication, division, and squaring. Use trial-and-error to deduce the rule, then apply it to solve the final test case.\n\n"
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `search`, `action`, or `answer`.\n\n"
                            "- Action Description:\n"
                            "  * `search`: If you choose `search`, you must only provide 'test' in the `content` field, indicating that you want to get the test case. All other contents are invalid, and search is only intended for retrieving the test case.\n"
                            "  * `action`: If you choose `action`, you must provide four numbers separated by spaces in the `content` field. All the provided numbers must be integers.\n"
                            "  * `answer`: If you choose `answer`, you must provide a single numerical answer in the `content` field. This answer should be the result of applying your guessed hidden function to the retrieved test case.\n\n"
                            "- Important Notes:\n"
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n"
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly choose the trial-and-error strategy, so that you can guess out the mapping function in the most efficient way.\n"
                            "  * Usually you should start your trials with small numbers. In each attempt, vary only one input at a time to isolate its effect on the output. Pay attention to how different positions might be prioritized: the function may change the order of inputs during calculation (e.g., operating on the 2nd and 3rd numbers first). Use comparisons between outputs to infer the underlying mapping rule, then apply your hypothesis to solve the final test case. You can provide your answer at any time and multiple times after retrieving the test case.\n"
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
                        ),
                },
                {
                    "role": "user",
                    "content": (
                            "My mapping function is settled and ready. You may begin your guessing and trials now!\n"
                    ),
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": expected_result, "env_name": "FunctionGym", "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "FunctionGym", "id": id},
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
