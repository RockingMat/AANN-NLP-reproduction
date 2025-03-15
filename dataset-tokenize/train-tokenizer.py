from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


DATA_FILE = "babylm-100m/train.sents"
NAME = "normal"


"""dataset = load_dataset(f"{NAME}-data", split="train")

def get_training_corpus():
    for example in dataset:
        yield example["text"]"""

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=16384,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

#tokenizer.train_from_iterator(get_training_corpus(), trainer)
files = [DATA_FILE]
tokenizer.train(files, trainer)

tokenizer.save(f"{NAME}-tokenizer.json")

print(f"Tokenizer training complete and saved as {NAME}-tokenizer.json")
