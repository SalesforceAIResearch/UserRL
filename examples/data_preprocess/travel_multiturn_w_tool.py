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
Preprocess the TravelGym dataset to parquet format
"""

import argparse
import os
import re
import json
from datasets import Dataset
import numpy as np
from verl.utils.hdfs_io import copy, makedirs


def main(wanted_num, one_choice_per_aspect):

    local_dir = f"/fsx/home/cqian/projects/dataset/travel{wanted_num}_multiturn"
    if one_choice_per_aspect:
        local_dir += "_onechoice"

    data_source = f"/fsx/home/cqian/projects/TravelGym/travelgym/data/travelgym_data_{wanted_num}.json"
    dataset = json.load(open(data_source))
    dataset_len = len(dataset)
    
    all_data = []
    for key, value in dataset.items():
        value["id"] = key
        all_data.append(value)
    
    # let the value to form a list
    dataset = list(all_data)
    np.random.shuffle(dataset)
    
    train_dataset = dataset[:int(dataset_len * 0.85)]
    test_dataset = dataset[int(dataset_len * 0.85):]
    
    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        id = example.pop("id")

        best_ids = []
        correct_ids = []
        dimensions = example["dimensions"]
        for dimension in dimensions:
            dim_data = example[dimension]
            best_ids.append(dim_data["best_id"])
            correct_ids.extend(dim_data["correct_ids"])
        initial_description = example["initial_description"]

        different_sentence = "  * Usually you should start by performing a search, then take action to actively uncover the user's preferences or reason to provide an answer. Keep in mind that multiple travel aspects require answers, and you are allowed to recommend only one option per aspect. Therefore, before making a recommendation, ensure you have thoroughly communicated with the user to understand their preferences.\n" if one_choice_per_aspect else \
        "  * Usually you should start by performing a search, then take action to actively uncover the user's preferences or reason to provide an answer. Keep in mind that multiple travel aspects require answers, and while you may answer multiple times, each answer should include only one option ID.\n"

        data = {
            "data_source": "interact_travelgym",
            "prompt": [
                {
                    "role": "system",
                    "content":
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n" +
                            "- Environment Description: TravelGym is an environment where you interact with both a user and a search database to fulfill a travel plan. Since the user's initial intent may be incomplete, you must proactively elicit preferences, perform searches, and make informed recommendations.\n\n" +
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `search`, `action` or `answer`.\n\n" +
                            "- Action Description:\n" +
                            "  * `search`: If you choose `search`, you must issue a clear and detailed query to the database in the `content` field. Specify the travel aspect you are searching for (e.g., hotel, flight, etc.) and provide well-supported arguments for your query. Only make one focused search attempt at a time.\n" +
                            "  * `action`: If you choose `action`, you will communicate directly with the user through the message you write in the `content` field. Your goal is to understand the user's preferences and intent by asking clear, specific questions. Avoid vague or overly general inquiries, and focus on detailed aspects of their travel needs.\n" +
                            "  * `answer`: If you choose `answer`, you must recommend a specific option to the user in the `content` field. Please only write one option ID from the database that is clearly tied to a particular travel aspect.\n\n" +
                            "- Important Notes:\n" +
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n" +
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly balance the number of rounds that you search, take action, or provide answer, so that you can fulfill the user's travel preferences in the most efficient way.\n" +
                            # Ensure the prompt reflects the one_choice_per_aspect setting
                            different_sentence +
                            "  * For each travel aspect, the user may have multiple preferences. What you ask may not directly align with the user's actual preferences, so you must proactively uncover them. Moreover, user preferences are often expressed implicitly, requiring careful interpretation.\n" +
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!",
                },
                {
                    "role": "user",
                    "content": (
                            f"{initial_description}\nAlso my budget is limited so as long as my preferences are satisfied, I would also like to choose the cheapest option for each.\n"
                    ),
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": str(best_ids), "env_name": "TravelGym", "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "TravelGym", "id": id},
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
    
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

if __name__ == "__main__":
    for wanted_num in ["2222", "444", "334", "333", "233", "44", "22", "33"]:
        for one_choice_per_aspect in [False, True]:
            np.random.seed(42)
            print(f"Processing wanted_num: {wanted_num}, one_choice_per_aspect: {one_choice_per_aspect}")
            main(wanted_num, one_choice_per_aspect)