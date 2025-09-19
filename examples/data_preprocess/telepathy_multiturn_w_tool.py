"""
Preprocess the Telepathy dataset to parquet format
"""

import argparse
import os
import re
import json
from datasets import Dataset
import numpy as np

np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/telepathy_multiturn")

    args = parser.parse_args()

    data_source = "./gyms/TelepathyGym/telepathygym/data/all_entities_refined.json"
    dataset = json.load(open(data_source))
    dataset_len = len(dataset)
    
    # let the value to form a list
    np.random.shuffle(dataset)
    
    train_dataset = dataset[:int(dataset_len * 0.9)]
    test_dataset = dataset[int(dataset_len * 0.9):]
    
    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        """
        {
        "id": "entity_502",
        "title": "Drone",
        "description": "I am thinking of a modern gadget. Try to guess what it is by asking yes/no questions!",
        "goal": "Guess the modern gadget I am thinking of.",
        "target_entity": "Drone - an unmanned aerial vehicle controlled remotely or autonomously, used for photography, surveillance, delivery, and recreational purposes.",
        "category": "gadget"
        }
        """
        title = example.pop("title")
        description = example.pop("description")
        goal = example.pop("goal")
        target_entity = example.pop("target_entity")
        category = example.pop("category")

        data = {
            "data_source": "interact_telepathygym",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n"
                            "- Environment Description: TelepathyGym is a guessing game where you are given a brief description of an entity, and your goal is to guess the entity. You should ask Yes-or-No questions to the environment to get more information about the entity.\n\n"
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `action` or `answer`.\n\n"
                            "- Action Description:\n"
                            "  * `action`: If you choose `action`, you must provide a Yes-or-No question in the `content` field to interact with the environment. Use this action to gather more information about the entity.\n"
                            "  * `answer`: If you choose `answer`, you must provide the entity in the `content` field that represents the most plausible guess based on your proposed questions and the corresponding environment responses.\n\n"
                            "- Important Notes:\n"
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n"
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly balance the number of rounds that you take action or provide answer, so that you can figure out the entity in the most efficient way.\n"
                            "  * Usually you should first take action to ask questions to gather more information about the entity. Then, provide your answer to validate your guess. You can provide your answer at any time and multiple times.\n"
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
                        ),
                },
                {
                    "role": "user",
                    "content": (
                            f"{description}\n"
                            "Try to uncover what's the entity I am thinking of!\n"
                    ),
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": target_entity, "env_name": "TelepathyGym", "title": title, "category": category},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "TelepathyGym", "title": title, "category": category},
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
