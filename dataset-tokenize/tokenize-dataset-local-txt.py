from datasets import load_dataset
from transformers import AlbertTokenizer

DATA_DIR = "babylm-100m"
NAME = "normal"

data_files = {
    "train": f"{DATA_DIR}/train.sents",
    "dev": f"{DATA_DIR}/dev.sents",
    "test": f"{DATA_DIR}/test.sents"
}

dataset = load_dataset("text", data_files=data_files)

# Load the tokenizer (you can choose one that fits your model)
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Define a tokenization function
def tokenize_function(example):
    # Tokenize the text field; adjust parameters as needed (e.g., truncation, padding)
    return tokenizer(example["text"], truncation=True)

# Apply the tokenizer to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset.set_format("torch")

tokenized_dataset.save_to_disk(f"{NAME}-data")
