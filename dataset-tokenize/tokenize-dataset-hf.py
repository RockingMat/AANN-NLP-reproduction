from datasets import load_dataset
from transformers import AlbertTokenizer

DATA_DIR = "kanishka/counterfactual_babylm_aann_indef_articles_with_pl_nouns_removal_new"
NAME = "noindefnns"

dataset = load_dataset(DATA_DIR)

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
