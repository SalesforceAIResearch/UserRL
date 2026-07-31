"""
Preprocess the SWEGym dataset to parquet format

Tasks originate from SWE-Together (https://arxiv.org/pdf/2606.29957), exported
into gyms/SWEGym/swegym/data by that benchmark's export_userrl_dataset.py.
"""

import argparse
import os
import json
from datasets import Dataset
import numpy as np

np.random.seed(42)


SYSTEM_PROMPT = (
    "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n"
    "- Environment Description: SWEGym replays a real software engineering session. A user has reported a problem in their codebase and you are the coding agent working on it. The user knows what they actually want, but their opening message leaves a lot unsaid: constraints, the shape of the fix they expect, and extra requirements they will only mention when they become relevant. Talk to them to draw this out, then submit the change you would make.\n\n"
    "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `action` or `answer`. Note that `search` is NOT available in this environment.\n\n"
    "- Action Description:\n"
    "  * `action`: If you choose `action`, you must provide a message to the user in the `content` field. Use it to ask a clarifying question, confirm an assumption, or describe the approach you are about to take. The user will reply as themselves.\n"
    "  * `answer`: If you choose `answer`, you must describe the concrete code change you are submitting in the `content` field: which files and functions you would modify and what the modification is. This is graded against the requirements of the task, so be specific about the change rather than restating the problem.\n\n"
    "- Important Notes:\n"
    "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n"
    "  * The total number of rounds that you can interact with the environment is limited. A question only pays off when it surfaces something the user actually cares about, so ask about one thing at a time and do not ask about what they have already told you.\n"
    "  * You may `answer` at any time and as many times as you like. Only your best submission counts, so it is safe to submit early and refine, but a submission that misses requirements you never asked about will score poorly.\n"
    "  * Be bold, creative and smart in your interaction with the environment! Let's begin!"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/swe_multiturn")

    args = parser.parse_args()

    data_dir = "./gyms/SWEGym/swegym/data"
    train_dataset = list(json.load(open(os.path.join(data_dir, "swe_tasks_train.json"))).values())
    test_dataset = list(json.load(open(os.path.join(data_dir, "swe_tasks_test.json"))).values())

    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")

    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        id = example["id"]
        instruction = example["instruction"]

        data = {
            "data_source": "interact_swe",
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": instruction,
                },
            ],
            "ability": "interaction",
            "reward_model": {"style": "rule", "ground_truth": id, "env_name": "SWEGym", "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "SWEGym", "id": id},
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
