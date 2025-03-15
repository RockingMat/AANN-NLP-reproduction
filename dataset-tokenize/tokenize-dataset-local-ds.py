from datasets import load_from_disk
from transformers import AlbertTokenizer

DATA_DIR = "pos-normal-raw"
NAME = "pos-normal"

dataset = load_from_disk(DATA_DIR)

# Load the tokenizer (you can choose one that fits your model)
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Define a tokenization function
def tokenize_function(example):
    # Tokenize the text field; adjust parameters as needed (e.g., truncation, padding)
    return tokenizer(example["pos_tags"], truncation=True)

# Apply the tokenizer to the dataset
tokenized_dataset = dataset.map(tokenize_function, remove_columns=["text", "pos_tags"], batched=True)

print(f"Tokenized dataset column namesa: {tokenized_dataset.column_names}")
print(f"Example train: {tokenized_dataset['train'][0]}")

tokenized_dataset.set_format("torch")

tokenized_dataset.save_to_disk(f"{NAME}-data")
