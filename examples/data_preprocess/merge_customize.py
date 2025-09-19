import os
import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
import yaml

tokenizer = AutoTokenizer.from_pretrained("/path/to/your/tokenizer")
pad_id = tokenizer.pad_token_id
print("Tokenizer Loaded!!!")

def load_dataset_split(dataset_name, split="train"):
    """Load a specific split from a dataset."""
    base_dir = f"./data/{dataset_name}"
    parquet_path = os.path.join(base_dir, f"{split}.parquet")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    dataset = Dataset.from_parquet(parquet_path)
    # turn into list
    dataset = dataset.to_list()
    return dataset

def merge_datasets(dataset_names, output_dir, split="train", shuffle=True):
    """Merge multiple datasets for a specific split."""
    datasets_to_merge = []
    tools = [yaml.safe_load(open("./examples/sglang_multiturn/config/tool_config/interact_tool_config.yaml"))["tools"][0]["tool_schema"]]
    
    all_prompt_token_lengths = []

    for dataset_name in dataset_names:
        dataset = load_dataset_split(dataset_name, split)
        if split == "train":
            if "travel" in dataset_name:
                np.random.shuffle(dataset)  # Shuffle travel datasets
                dataset = dataset[:int(len(dataset) * 0.35)]
            if "function" in dataset_name:
                np.random.shuffle(dataset)  # Shuffle travel datasets
                dataset = dataset[:200]
        elif split == "test":
            if "travel" in dataset_name:
                np.random.shuffle(dataset)  # Shuffle travel datasets
                dataset = dataset[:int(len(dataset) * 0.5)]
            else:
                min_number = max(50, int(len(dataset) * 0.5))
                np.random.shuffle(dataset)  # Shuffle test datasets
                dataset = dataset[:min_number]

        prompt_token_lengths = []
        discarded = 0
        for sample in dataset:
            messages = sample["prompt"]
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, tools=tools, add_generation_prompt=True)
            if len(prompt) >= 1152:
                discarded += 1
            else:
                datasets_to_merge.append(sample)
            prompt_token_lengths.append(len(prompt))
        
        print(f"Dataset {dataset_name} - {split} split: {len(dataset)} samples")

    np.random.shuffle(datasets_to_merge)
    merged_dataset = Dataset.from_list(datasets_to_merge)
    print(f"Merged {split} dataset: {len(merged_dataset)} total samples")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}.parquet")
    merged_dataset.to_parquet(output_path)
    print(f"Saved merged {split} dataset to {output_path}")

    return merged_dataset


def main():
    np.random.seed(42)
    # Merge train split
    merge_datasets(
        ["travel22_multiturn_onechoice", "travel33_multiturn_onechoice", "travel44_multiturn_onechoice", "travel233_multiturn_onechoice", "travel333_multiturn_onechoice", "travel334_multiturn_onechoice", "travel444_multiturn_onechoice", "travel2222_multiturn_onechoice", "turtle_multiturn", "persuasion_multiturn", "function_multiturn", "tau_multiturn"], 
        output_dir=f"./data/alltrain_multiturn",
        split="train"
    )

    np.random.seed(42)
    # Merge test split
    merge_datasets(
        ["travel22_multiturn_onechoice", "travel33_multiturn_onechoice", "travel44_multiturn_onechoice", "travel233_multiturn_onechoice", "travel333_multiturn_onechoice", "travel334_multiturn_onechoice", "travel444_multiturn_onechoice", "travel2222_multiturn_onechoice", "turtle_multiturn", "telepathy_multiturn", "persuasion_multiturn", "intention_multiturn", "bamboogle_multiturn", "function_multiturn", "tau_multiturn"], 
        output_dir=f"./data/alltest_multiturn",
        split="test"
    )

if __name__ == "__main__":
    main()