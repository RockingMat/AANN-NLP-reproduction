import torch
from transformers import LlamaForCausalLM, AlbertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "pos-normal-model"

model = LlamaForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", padding_side="left") # set padding_side="left" is important for open ended text generation

model.generation_config.pad_token_id = tokenizer.pad_token_id

input_text = "Today I cooked a piece of"
print("INPUT: " + input_text)
tokenized_input = tokenizer(input_text, return_tensors="pt").to(device)
del tokenized_input["token_type_ids"]

with torch.no_grad():
    for _ in range(5):
        output_ids = model.generate(
            **tokenized_input,
            max_length=256,
            do_sample=True#,
            #top_k=50,
            #temperature=0.7
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        print("\nOUTPUT: " + output_text)
