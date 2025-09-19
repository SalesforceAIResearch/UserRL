import json
import os
import argparse
import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("/fsx/home/cqian/projects/model/Qwen2.5-3B-Instruct")
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

def merge_datasets(dataset_names, split="train", shuffle=True):
    """Merge multiple datasets for a specific split."""
    datasets_to_merge = []
    
    for dataset_name in dataset_names:
        dataset = load_dataset_split(dataset_name, split)
        datasets_to_merge.extend(dataset)
    
    # Shuffle if requested
    if shuffle:
        np.random.shuffle(datasets_to_merge)
    
    if split == "train":
        # get the prompt token length (chat template to tokenize)
        filtered_samples = []
        prompt_token_lengths = []
        for sample in datasets_to_merge:
            messages = sample["prompt"]
            prompt = tokenizer.apply_chat_template(messages, tokenize=True)
            prompt_token_lengths.append(len(prompt))
            if len(prompt) <= 600:
                filtered_samples.append(sample)

        # Draw the distribution of prompt token length
        draw_prompt_token_length_distribution(prompt_token_lengths)

        print(f"Prompt token length top 10: {np.sort(prompt_token_lengths)[-10:]}")
        print(f"Prompt token length mean: {np.mean(prompt_token_lengths)}")
        print(f"Prompt token length std: {np.std(prompt_token_lengths)}")
        print(f"Prompt token length max: {np.max(prompt_token_lengths)}")
        print(f"Before filtering: {len(datasets_to_merge)}")
        print(f"Filtered samples: {len(filtered_samples)}")
    
        datasets_to_merge = filtered_samples

    merged_dataset = Dataset.from_list(datasets_to_merge)
    
    print(f"Merged {split} dataset: {len(merged_dataset)} total samples")
    return merged_dataset

def save_merged_dataset(merged_train, merged_test, output_dir):
    """Save merged train and test datasets to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    merged_train.to_parquet(train_path)
    merged_test.to_parquet(test_path)
    
    print(f"Saved merged datasets: train({len(merged_train)}) test({len(merged_test)})")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple interaction datasets")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["turtle_multiturn", "telepathy_multiturn", "persuasion_multiturn", "intention_multiturn"],
        help="List of datasets to merge"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--no_shuffle", 
        action="store_true",
        help="Don't shuffle the merged dataset"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Always save to interact_multiturn directory
    output_dir = "/fsx/home/cqian/projects/dataset/interact_multiturn"
    
    # Merge train and test splits
    merged_train = merge_datasets(
        args.datasets, 
        split="train", 
        shuffle=not args.no_shuffle, 
    )
    
    merged_test = merge_datasets(
        args.datasets, 
        split="test", 
        shuffle=not args.no_shuffle, 
    )
    
    # Save merged datasets
    save_merged_dataset(merged_train, merged_test, output_dir)

if __name__ == "__main__":
    main()