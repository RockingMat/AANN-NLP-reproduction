# pip install transformers datasets torch

import os
import torch
from datasets import load_from_disk
from transformers import (
    LlamaConfig, LlamaForCausalLM, 
    DataCollatorForLanguageModeling,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    PreTrainedTokenizerFast
)
# from huggingface_hub import login

# # Load Hugging Face Token
# hf_token = os.getenv("HF_TOKEN")
# if hf_token is None:
#     raise ValueError("HF_TOKEN not set. Please set your Hugging Face token in environment variables.")
# login(token=hf_token)



NAME = "tok-normal-10m" # !!!!! this name will be applied to load data, save model checkpoints, and the actual model



# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
dataset_path = f"./{NAME}-data"  # Adjust path if needed
tokenized_datasets = load_from_disk(dataset_path)
print(f"Using data: {dataset_path}")

# Model configuration
config = LlamaConfig(
    vocab_size=16384,               # Different from "Vocab size = 16384" in the paper (and different tokenizer too)
    hidden_size=512,                # Different from "Embed size = 768" in the paper
    intermediate_size=2048,         # Different from "FFN dimension = 3072" in the paper
    num_attention_heads=12,         # Corresponds to "Attention heads = 12" in the paper
    num_hidden_layers=6,            # Different from "Num. layers = 12" in the paper
    max_position_embeddings=256,    # Corresponds to AlbertTokenizer, different from "Max. seq. length = 256" in the paper
    use_cache=False
)

# Initialize model
model = LlamaForCausalLM(config).to(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="normal-tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    sep_token="[SEP]",
    cls_token="[CLS]",
    mask_token="[MASK]"
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{NAME}-model-checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,                         # Different from "Epochs = 20" in the paper
    learning_rate=3e-4,
    per_device_train_batch_size=32,             # Corresponds to "Batch size = 32" in the paper
    per_device_eval_batch_size=32,              # Different from Kanishka's HuggingFace of 64
    lr_scheduler_type="linear",
    warmup_steps=32000,                         # Corresponds to "Warmup steps = 32000" in the paper
    logging_steps=50,
    logging_dir="logs",
    save_total_limit=2,
    gradient_accumulation_steps=1,              # Different from Kanishka's HuggingFace of 8
    fp16=True if device == "cuda" else False,
    seed=42
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save final model
trainer.save_model(f"{NAME}-model")

# Optional: Push to Hugging Face Hub
# trainer.push_to_hub("your-username/model-name")
