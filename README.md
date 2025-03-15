# AANN-NLP-reproduction
All code used to reproduce https://aclanthology.org/2024.emnlp-main.53.pdf


# Reproduction steps

1. Get data using bash ./data/get_babylm1.sh
2. Preprocess data with Albert Tokenizer: Yuchen add dataset factory details here
3. Preprocess Morph data by running all notebooks in morph folder(use aann_corruption.csv for morph_enc_no_aann and use train.sents for morph_enc_unablated)
4. Train models: Use train scripts: Yuchen
5. Train Unigram models by running unigram.ipynb under unigram model. All of the trained models are in the folder.
6. Compute slor scores: Use Slor Scripts: Yuchen
