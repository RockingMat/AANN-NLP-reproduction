from datasets import load_from_disk, DatasetDict

# Paths to your dataset splits
train_dataset_path = "train"
val_dataset_path = "validation"

# Load the datasets
train_dataset = load_from_disk(train_dataset_path)
val_dataset = load_from_disk(val_dataset_path)

# Combine into a DatasetDict
full_dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Save the merged dataset to a new directory
merged_dataset_path = "pos-normal-data"
full_dataset.save_to_disk(merged_dataset_path)

print(f"Merged dataset saved to {merged_dataset_path}")
