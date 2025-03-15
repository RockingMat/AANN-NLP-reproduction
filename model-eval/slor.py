import torch
from transformers import LlamaForCausalLM, AlbertTokenizer
import numpy as np

class UnigramLM:
    def __init__(self, counts_path):
        self.counts = {}
        self.total_counts = 0
        self.load_counts(counts_path)

    def load_counts(self, counts_path):
        with open(counts_path, "r") as f:
            for line in f:
                word, prob = line.strip().split()
                self.counts[word] = float(prob)
        self.total_counts = sum(self.counts.values())

    def get_probability(self, token):
        return self.counts.get(token, 1e-8)

def get_lm_probability(prefix, construction):
    prefix_ids = tokenizer.encode(prefix, return_tensors="pt")
    output_ids = tokenizer.encode(construction, return_tensors="pt")
    full_ids = torch.cat([prefix_ids, output_ids], dim=-1).to(device)

    with torch.no_grad():
        outputs = model(full_ids)
    logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    prefix_length = prefix_ids.shape[1]
    for i in range(output_ids.shape[1]):
        # The prediction for output token i is at position (prefix_length + i - 1)
        token_id = output_ids[0, i]
        token_log_prob = log_probs[0, prefix_length + i - 1, token_id]
        total_log_prob += token_log_prob.item()

    return np.exp(total_log_prob)

def get_unigram_probability(construction):
    tokens = tokenizer.tokenize(construction)
    probs = [unigram_model.get_probability(token) for token in tokens]
    return np.exp(sum(np.log(probs)))

def compute_slor(prefix, construction):
    pm = get_lm_probability(prefix, construction)
    pu = get_unigram_probability(construction)
    length = max(1, len(tokenizer.tokenize(construction)))
    return np.log(pm / pu) / length

# ------------------------------------------------------------------------ #

NAME = "tok-normal-10m"
model_path = f"{NAME}-model"
unigram_model = UnigramLM(f"{NAME}-unigram.txt")

# ------------------------------------------------------------------------ #

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
model = LlamaForCausalLM.from_pretrained(model_path).to(device)
model.eval()

prefixes = ["Yesterday I trained", "This week has been", "In"]
correct = ["a whopping ninety LMs", "a beautiful five days", "the last five months"]
corruptions = [
    ["a ninety whopping LMs", "a five beautiful days", "the five last months"], # order-swap
    ["whopping ninety LMs", "beautiful five days", "last five months"], # no article
    ["a ninety LMs", "a five days", "the five months"], # no modifier
    ["a whopping LMs", "a beautiful days", "the last months"]  # no numeral
]

for i, prefix in enumerate(prefixes):
    sent = correct[i]
    c_sents = [version[i] for version in corruptions]
    print(f"Prefix: {prefix}")
    print(f"Correct: {sent}")
    print(f"Corruptions: {c_sents}")
    slor_correct = compute_slor(prefix, sent)
    slor_corrupt = [compute_slor(prefix, c_sent) for c_sent in c_sents]
    print(f"SLOR (Correct): {slor_correct}")
    print(f"SLOR (Corrupt): {slor_corrupt}")
    print("\n")

print("\nDone with explicit testing. Now executing on HF dataset.\n\n")

import pandas as pd
from tqdm import tqdm

ds = pd.read_csv("aann_corruption.csv")

num_correct = 0
avg_correct = 0
avg_order_swap = 0
avg_no_article = 0
avg_no_modifier = 0
avg_no_numeral = 0
total = 0
for index, row in tqdm(ds.iterrows(), total=ds.shape[0]):
    prefix = row["prefixes"]
    slor_correct = compute_slor(prefix, row["construction"])
    slor_order_swap = compute_slor(prefix, row["order_swap"])
    slor_no_article = compute_slor(prefix, row["no_article"])
    slor_no_modifier = compute_slor(prefix, row["no_modifier"])
    slor_no_numeral = compute_slor(prefix, row["no_numeral"])
    if slor_correct > slor_order_swap and slor_correct > slor_no_article and slor_correct > slor_no_modifier and slor_correct > slor_no_numeral:
        num_correct += 1
    avg_correct += slor_correct
    avg_order_swap += slor_order_swap
    avg_no_article += slor_no_article
    avg_no_modifier += slor_no_modifier
    avg_no_numeral += slor_no_numeral
    total += 1

print(f"Got {num_correct} correct out of {total} for an accuracy of {num_correct / total}.")
avg_correct /= total
avg_order_swap /= total
avg_no_article /= total
avg_no_modifier /= total
avg_no_numeral /= total
print(f"Average SLORS:\n\tCorrect: {avg_correct}\n\tOrder swap: {avg_order_swap}\n\tNo article: {avg_no_article}\n\tNo modifier: {avg_no_modifier}\n\tNo numeral: {avg_no_numeral}")
