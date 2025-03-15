# AANN-NLP-reproduction
By Yuchen Xin, Yash Mathur, and Hyeonjeong Byeon.

This repository contains data/code to reproduce the [AANN paper](https://aclanthology.org/2024.emnlp-main.53.pdf) written by Dr. Kanishka Misra and Dr. Kyle Mahowald, as well as for additional experiments described in our paper.

## Set-Up and Dependencies
Replicate the environment by creating a conda environment with `conda env create -f environment.yml`,  using the provided `environment.yml` file in the base directory of the repository.

## Reproduction Steps
### Data Download
Call `bash ./data_download/get_babylm1.sh` to download the public BabyLM 100M/10M corpus into the specified `$DIR` within the script.

### Preprocessing
Refer to the `dataset-tokenize` directory, which contains scripts to preprocess the data by passing them through a tokenizer.

If using the pretrained `albert-base-v2` Albert Tokenizer from the `transformers` library of HuggingFace to tokenize the data, invoke `python dataset-tokenize/SCRIPT` where `SCRIPT` could be any of the following (depending on use case):

- tokenize-dataset-hf.py: download a dataset from HuggingFace and tokenize it, saving to disk
- tokenize-dataset-local-ds.py: load a local HuggingFace-format dataset and tokenize it.
- tokenize-dataset-local-txt.py: tokenize a local data directory with the files `train.sents`, `dev.sents`, and `test.sents`.

If training a BPE tokenizer on specified data, invoke `python dataset-tokenize/train-tokenizer.py` after modifying the script to access the right data file, which will save a `NAME-tokenizer.json` file. Then use that file to tokenize data by calling `python tokenize-custom.py`.

### Training
To train autoregressive transformer language models, do one of the following:
- If using `albert-base-v2` as the tokenizer, invoke `python model-train/train-lm.py`.
- If using custom BPE tokenizer, copy the `NAME-tokenizer.json` file over and run `python model-train/train-lm-custom.py`.

**Adjust hyperparameters within these scripts as necessary!**

To train unigram models for SLOR score computation in the evaluation step, import the relevant training data files (in text, before tokenization) and run the `unigram/unigram.ipynb` Jupyter Notebook. Otherwise, use pre-generated `unigram/unigram-models` which are all saved as `.txt` files.

### Evaluation
When training completes, the models will be saved in the HuggingFace format locally, and will contain files stating train loss, eval loss, etc.

Before proceeding with SLOR evaluation, ensure that the HuggingFace transformers model is saved locally, and so is the unigram model `.txt` file. You can also check the transformer LMs are working properly by running `python model-eval/test-lm.py` which will do some open-ended text generation on some specified prefix.

Use the `data_download/aann_corruption.csv` file (obtained from [Mahowald's repository](https://github.com/kanishkamisra/aannalysis/tree/main/data/mahowald-aann)) which contains all prefixes, correct AANN constructions, and counterfactual corruptions. Then invoke `python model-eval/SCRIPT` where `SCRIPT` is one of the following:

- slor.py: for models trained on `albert-base-v2` Albert Tokenizer.
- slor-tok.py: for models trained on a custom BPE tokenizer.

The final results will be printed to the terminal.

## Additional Experiment Steps
To train a model with POS tags appended, refer to the `morph` directory. Run the corresponding notebooks in this directory to generate the data, such as

- `morph_enc_local.ipynb`: load a dataset locally and append POS tags to each word after a pipe "|", saving locally.
- `morph_enc_hf.ipynb`: load a dataset from HuggingFace and append POS tags to each word after a pipe "|", saving locally.
- `morph_enc_slor.ipynb`: transform the eval dataset (e.g. `aann_corruption.csv`) into one with appended POS tags. Necessary for evaluation of models trained on datasets with POS-tag-appending.

Then, follow the same training and evaluation steps as before.