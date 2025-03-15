# AANN-NLP-reproduction
All code used to reproduce https://aclanthology.org/2024.emnlp-main.53.pdf


# Reproduction steps

1. Create a conda environment using following script: conda env create -f environment.yml
2. Get data using bash ./data/get_babylm1.sh
3. Preprocess data with Albert Tokenizer: Yuchen add dataset factory details here
4. Preprocess Morph data by running all notebooks in morph folder(use aann_corruption.csv for morph_enc_no_aann and use train.sents for morph_enc_unablated)
5. Train models: Use train scripts: Yuchen
6. Train unigram models by running unigram.ipynb in unigram folder. All of the trained models are also included.
7. Compute slor scores: Use Slor Scripts: Yuchen
