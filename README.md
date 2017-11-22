# PCA-based Relation Extraction
This is the code release for the method described in the blog post [A Fast Baseline for Sentence Representation Learning (Spoiler Alert: It’s PCA and Logistic Regression)](https://hazyresearch.github.io/snorkel/blog/pca_lstm)


## Installation

Our experiments are built using [Snorkel](https://hazyresearch.github.io/snorkel/), Stanford's data creation and management systems, and a lightweight experimental framework for conducting model search.

### Snorkel Dependencies
If you already have Snorkel and [PyTorch](http://pytorch.org/) running, you can skip this section. 

Setup is easiest if you install [miniconda](https://conda.io/miniconda.html) and create a custom enviornment for Python 2.7

```
conda create -n py27 python=2.7
source activate py27
```
Then install all Snorkel dependencies

```
./install_deps.sh
```

Install PyTorch for Linux (see [here](http://pytorch.org/) for OSX)

```
conda install pytorch torchvision cuda80 -c soumith
```

### Installing PCA/LSTM Code
 
Once all Snorkel dependencies are installed, run 

```
./install.sh
```
Snorkel requires a `SNORKELHOME` enviornment variable pointing to the parent directory of your install. On Linux/OSX, you can manuallly add this to your bash setup by editing `.bashrc` or `.bash_profile`. Otherwise, just run this script to temporarily setup your enviornment.

```
source ./set_env.sh
```


## Download Embeddings
The PCA-based method requires pre-trained word embeddings. We trained our own embeddings with Wikipedia and PubMed data using [FastText](https://github.com/facebookresearch/fastText) and Gensim's implementation of [word2vec](https://radimrehurek.com/gensim/models/word2vec.html). You can download our embeddings using:
`./download.sh` (7.7GB compressed).

You can use also any public embedding datasets:

- [GloVe](https://nlp.stanford.edu/projects/glove/) Common Crawl
- [FastText](https://fasttext.cc/docs/en/english-vectors.html) Wiki-News / Common Crawl
- [PubMed Biomedical](<https://drive.google.com/open?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8>) (Chiu et al. 2016)

When running models, just change the `word_emb_path` parameter to point to your embeddings. 

## Quick Start

The following datasets are included here (see the blog for more details). They were generated using the Snorkel tutorials found [here](https://github.com/HazyResearch/snorkel/tree/master/tutorials).

- `cdr-supervised` 
- `cdr-dp` 
- `spouse-supervised`
- `spouse-db`

All BiLSTM and PCA model configurations are JSON files found in `config/`.

The flag `--debug` trains on a 10% subset of data and is useful to test your installation.

```
python benchmark.py -d cdr-supervised --config configs/lstm.json -E 50 --debug
```

## Training Models

### Train a basic BiLSTM

```
python benchmark.py -d cdr-supervised --config configs/lstm.json --n_model_search 1 -E 50
```

This trains a BiLSTM *without* pre-trained embeddings for 50 epochs. 

### Train a BiLSTM + attention model with FastText embeddings

You can override any of the parameters specified in the model JSON by passing in a string of of the format `<PARAM_NAME>=<VALUE>` seperated by a comma. 

```
python benchmark.py -d cdr-supervised --config configs/lstm.json -p attention=True,word_emb_dim=300,word_emb_path="data/embs/pubmed/fastText/300/pubmed.d300.w30.neg10.fasttext.vec" --n_model_search 1 -E 50
```

### Train a PCA + exp. decay model with FastText embeddings

```
python benchmark.py -d cdr-supervised --config configs/pca-kernel.json -p word_emb_dim=300,word_emb_path="data/embs/pubmed/fastText/300/pubmed.d300.w30.neg10.fasttext.vec" --n_model_search 1 -E 50
```

## Model Search
Our config files contain the same hyperparameter search space used in our experiments, just pass in a value >1 for `n_model_search`. 

```
python benchmark.py -d cdr-supervised --config configs/lstm.json --n_model_search 5 -E 50
```

We searched over 50 models for our inital experiments. The top 5 parameter configurations we found for BiLSTMs and PCA are in `config/top5/`.


## Running Experimental Benchmarks

You can replicate the core experiments described in our blog post by running the jobs from [script](https://github.com/HazyResearch/PCA-Relation-Extraction/blob/master/experiments.sh). 

**Warning**: this trains 80 models. We distributed our experiments across several machines and ran the BiLSTMs on GPUs. 


## Citations

If you use this work in any way, please cite us as:

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
