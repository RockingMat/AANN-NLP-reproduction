from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

DATA_DIR = "babylm-10m"
NAME = "normal-10m-tok"
TOKENIZER = "normal-tokenizer.json"

data_files = {
    "train": f"{DATA_DIR}/train.sents",
    "dev": f"{DATA_DIR}/dev.sents",
    "test": f"{DATA_DIR}/test.sents"
}

dataset = load_dataset("text", data_files=data_files)

# Load the tokenizer (you can choose one that fits your model)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER,
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    mask_token="[MASK]"
)

# Define a tokenization function
def tokenize_function(example):
    # Tokenize the text field; adjust parameters as needed (e.g., truncation, padding)
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256, return_tensors="pt")

# Apply the tokenizer to the dataset
tokenized_dataset = dataset.map(tokenize_function, remove_columns=["text"], batched=True, num_proc=4)

tokenized_dataset.set_format("torch")

tokenized_dataset.save_to_disk(f"{NAME}-data")
