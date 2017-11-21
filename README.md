
#PCA-based Relation Extraction
This is the code release for the method described in the blog post [A Fast Baseline for Sentence Representation Learning (Spoiler Alert: It’s PCA and Logistic Regression)](https://hazyresearch.github.io/snorkel/blog/pca_lstm)


## Installation

Our experiments are build using [Snorkel](), Stanford's data creation and management systems, and a lightweight experimental framework for conducting model search.

Installation is easiest if you download and install `conda`. 

`conda install snorkel`

The PCA-based method requires pre-trained word embeddings. You can use any of the standard public embedding datasets (e.g., [GloVe](https://nlp.stanford.edu/projects/glove/)). We custom trained embeddings from Wikipedia and PubMed using [FastText](https://github.com/facebookresearch/fastText) and [word2vec](https://radimrehurek.com/gensim/models/word2vec.html). 

You can download our exact embeddings using `./download_data.sh`

## Quick Start

`python benchmark.py --info` prints all available datasets.  All BiLSTM and PCA model configurations are JSON files found in `config/`.

The flag `--debug` trains on a 10% subset of data and is useful to test your installation.

`python benchmark.py -d cdr-supervised --config configs/lstm.json -E 50 --debug`

## Training Models

###Train a basic BiLSTM

`python benchmark.py -d cdr-supervised --config configs/lstm.json --n_model_search 1 -E 50`

This trains a BiLSTM *without* pre-trained embeddings for 50 epochs. 
You can override any of the parameters specified in the model JSON by passing in a string of of the format `<PARAM_NAME>=<VALUE>` seperated by a comma. 

###Train a BiLSTM + attention model with FastText embeddings

`python benchmark.py -d cdr-supervised --config configs/lstm.json -p attention=True,word_emb_dim=300,word_emb_path="data/embs/pubmed/fastText/300/pubmed.d300.w30.neg10.fasttext.vec" --n_model_search 1 -E 50`

###Train a PCA + Exp. Decay model with FastText embeddings

`python benchmark.py -d cdr-supervised --config configs/pca-kernel.json -p word_emb_dim=300,word_emb_path="data/embs/pubmed/fastText/300/pubmed.d300.w30.neg10.fasttext.vec" --n_model_search 1 -E 50`

## Model Search
Our config files contain the same hyperparameter search space used in our experiments, just pass in a value >1 for `n_model_search`. 

`python benchmark.py -d cdr-supervised --config configs/lstm.json --n_model_search 5 -E 50`

We searched over 50 models for our inital experiments. The top 5 parameter configurations we found for BiLSTMs and PCA are in `config/top5/`.


## Running Experimental Benchmarks

You can replicate the experiments described in the blog post by running the jobs from [script](). 

This trains 80 models. We distributed our experiments across several machines and ran the BiLSTMs on GPUs. 


##Citations

If you use this work, please cite us as:

```
Fries, J., Wu, S., Choi, K., Marsden, A., R{\'e}, C., "A Fast Baseline for Sentence Representation Learning (Spoiler Alert: It’s PCA and Logistic Regression)", HazyResearch, 2017. https://hazyresearch.github.io/snorkel/blog/pca_lstm
```

```
@article{fries2017pcalstm,
  author={Fries, Jason and Wu, Sen and Choi, Kristy and Marsden, Annie and R{\'e}, Christopher},
  title = {A Fast Baseline for Sentence Representation Learning (Spoiler Alert: It’s PCA and Logistic Regression)},
  journal = {HazyResearch},
  year = {2017},
  url = {https://hazyresearch.github.io/snorkel/blog/pca_lstm}
}
```

