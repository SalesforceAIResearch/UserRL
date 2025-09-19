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
    parser.add_argument("--local_dir", default="/fsx/home/cqian/projects/dataset/bamboogle_multiturn")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/fsx/home/cqian/projects/SearchGym/searchgym/data/questions.json"
    dataset = json.load(open(data_source))
    dataset_len = len(dataset)

    filtered_dataset = []
    for item in dataset:
        if "Bamboogle" in item["id"]:
            filtered_dataset.append(item)
    dataset = filtered_dataset
    dataset_len = len(dataset)
    
    # let the value to form a list
    np.random.shuffle(dataset)
    
    # train_dataset = dataset[:int(dataset_len * 0.9)]
    test_dataset = dataset
    
    print(f"test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        """
        {
        "id": "Bamboogle-1",
        "question": "Who was president of the United States in the year that Citibank was founded?",
        "answer": "james madison"
        }
        """
        question = example.pop("question")
        answer = example.pop("answer")
        id = example.pop("id")

        data = {
            "data_source": "interact_searchgym",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n"
                            "- Environment Description: SearchGym is a question answering environment where you're given a question and your goal is to provide the correct answer to it. You can search the web step by step to get more information about the question and then answer the question.\n\n"
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `search` or `answer`.\n\n"
                            "- Action Description:\n"
                            "  * `search`: If you choose `search`, you must propose a question in the `content` field to interact with the environment. The question should be a search query. You should use this action to try to get more information about the question step by step.\n"
                            "  * `answer`: If you choose `answer`, you must provide a short answer to the question based on your knowledge and the search results from earlier steps.\n\n"
                            "- Important Notes:\n"
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n"
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly choose your searching strategy, so that you can figure out the answer to the question in the most efficient way.\n"
                            "  * Usually you should make a plan about how to search for information step by step and then provide your answer. You can provide your answer at any time and multiple times.\n"
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
                        ),
                },
                {
                    "role": "user",
                    "content": (
                            f"{question}\n"
                            "Please make a plan to search for information step by step and provide your answer to my question!\n"
                    ),
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": answer, "env_name": "SearchGym", "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "SearchGym", "id": id},
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
