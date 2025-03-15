from transformers import AutoConfig

config = AutoConfig.from_pretrained("albert-base-v2")
print(config.max_position_embeddings)
print(config.vocab_size)
