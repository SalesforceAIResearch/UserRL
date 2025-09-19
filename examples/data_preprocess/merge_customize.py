import json
import os
import argparse
import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import yaml

tokenizer = AutoTokenizer.from_pretrained("/fsx/home/cqian/projects/model/Qwen3-8B")
pad_id = tokenizer.pad_token_id
print("Tokenizer Loaded!!!")

def draw_prompt_token_length_distribution(lengths):
    """Draw the distribution of prompt token length"""
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=10, color="skyblue", edgecolor="black")
    plt.xlabel("Prompt Token Length")
    plt.ylabel("Frequency")
    plt.title("Prompt Token Length Distribution")
    plt.savefig("prompt_token_length_distribution.png", dpi=300)
    plt.close()

def load_dataset_split(dataset_name, split="train"):
    """Load a specific split from a dataset."""
    base_dir = f"/fsx/home/cqian/projects/dataset/{dataset_name}"
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
    tools = [yaml.safe_load(open("/fsx/home/cqian/projects/verl/examples/sglang_multiturn/config/tool_config/interact_tool_config.yaml"))["tools"][0]["tool_schema"]]
    
    all_prompt_token_lengths = []

    for dataset_name in dataset_names:
        dataset = load_dataset_split(dataset_name, split)
        if split == "train":
            if "travel" in dataset_name:
                np.random.shuffle(dataset)  # Shuffle travel datasets
                dataset = dataset[:int(len(dataset) * 0.35)]
            # if "function" in dataset_name:
            #     np.random.shuffle(dataset)  # Shuffle travel datasets
            #     dataset = dataset[:200]
        # elif split == "test":
        #     if "travel" in dataset_name:
        #         np.random.shuffle(dataset)  # Shuffle travel datasets
        #         dataset = dataset[:int(len(dataset) * 0.5)]
        #     else:
        #         min_number = max(50, int(len(dataset) * 0.5))
        #         np.random.shuffle(dataset)  # Shuffle test datasets
        #         dataset = dataset[:min_number]

        # prompt_token_lengths = []
        # discarded = 0
        # for sample in dataset:
        #     messages = sample["prompt"]
        #     prompt = tokenizer.apply_chat_template(messages, tokenize=True, tools=tools, add_generation_prompt=True)
        #     if len(prompt) >= 1152:
        #         discarded += 1
        #     else:
        #         datasets_to_merge.append(sample)
        #     prompt_token_lengths.append(len(prompt))
            
        # prompt_token_lengths = np.sort(prompt_token_lengths)
        # all_prompt_token_lengths.extend(prompt_token_lengths)
        print(f"Dataset {dataset_name} - {split} split: {len(dataset)} samples")
        # print(f"Discarded {discarded} samples")
        # print(f"Prompt token length top 10: {prompt_token_lengths[-10:][::-1]}")
        # print(f"Prompt token length mean: {np.mean(prompt_token_lengths)}")
        # print(f"Prompt token length max: {np.max(prompt_token_lengths)}")
        # print(f"Prompt token length 90th percentile: {np.percentile(prompt_token_lengths, 90)}")
    
    draw_prompt_token_length_distribution(all_prompt_token_lengths)
    
    
    # if split == "train" or split == "test":
    #     # get the prompt token length (chat template to tokenize)
    #     filtered_samples = []
    #     prompt_token_lengths = []
    #     for sample in datasets_to_merge:
    #         messages = sample["prompt"]
    #         prompt = tokenizer.apply_chat_template(messages, tokenize=True)
    #         prompt_token_lengths.append(len(prompt))
    #         if len(prompt) <= 700:
    #             filtered_samples.append(sample)

    #     # Draw the distribution of prompt token length
    #     draw_prompt_token_length_distribution(prompt_token_lengths)

    #     print(f"Prompt token length top 10: {np.sort(prompt_token_lengths)[-10:][::-1]}")
    #     print(f"Prompt token length mean: {np.mean(prompt_token_lengths)}")
    #     print(f"Prompt token length max: {np.max(prompt_token_lengths)}")
    #     print(f"Before filtering: {len(datasets_to_merge)}")
    #     print(f"Filtered samples: {len(filtered_samples)}")
    
    #     datasets_to_merge = filtered_samples

    np.random.shuffle(datasets_to_merge)

    merged_dataset = Dataset.from_list(datasets_to_merge)
    
    print(f"Merged {split} dataset: {len(merged_dataset)} total samples")

    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, f"{split}.parquet")
    # merged_dataset.to_parquet(output_path)
    # print(f"Saved merged {split} dataset to {output_path}")

    return merged_dataset


def main():
    parser = argparse.ArgumentParser(description="Merge multiple interaction datasets")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["travel22_multiturn_onechoice", "travel33_multiturn_onechoice", "travel44_multiturn_onechoice", "travel233_multiturn_onechoice", "travel333_multiturn_onechoice", "travel334_multiturn_onechoice", "travel444_multiturn_onechoice", "travel2222_multiturn_onechoice", "turtle_multiturn", "telepathy_multiturn", "persuasion_multiturn", "intention_multiturn", "bamboogle_multiturn", "function_multiturn", "tau_multiturn"],
        help="List of datasets to merge"
    )
    # Train: ["travel22_multiturn_onechoice", "travel33_multiturn_onechoice", "travel44_multiturn_onechoice", "travel233_multiturn_onechoice", "travel333_multiturn_onechoice", "travel334_multiturn_onechoice", "travel444_multiturn_onechoice", "travel2222_multiturn_onechoice", "turtle_multiturn", "persuasion_multiturn", "function_multiturn", "tau_multiturn"],
    # Test: ["travel22_multiturn_onechoice", "travel33_multiturn_onechoice", "travel44_multiturn_onechoice", "travel233_multiturn_onechoice", "travel333_multiturn_onechoice", "travel334_multiturn_onechoice", "travel444_multiturn_onechoice", "travel2222_multiturn_onechoice", "turtle_multiturn", "telepathy_multiturn", "persuasion_multiturn", "intention_multiturn", "bamboogle_multiturn", "function_multiturn", "tau_multiturn"],
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Always save to interact_multiturn directory
    output_dir = f"/fsx/home/cqian/projects/dataset/all{args.split}_multiturn_new0910"
    
    # Merge train and test splits
    merge_datasets(
        args.datasets, 
        output_dir=output_dir,
        split=args.split
    )

if __name__ == "__main__":
    main()