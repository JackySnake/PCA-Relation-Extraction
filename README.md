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

### Installing PCA / BiLSTM Code
 
Once all Snorkel dependencies are installed, run 

```
./install.sh
```
Snorkel requires a `SNORKELHOME` environment variable pointing to the parent directory of your install. On Linux/OSX, you can manuallly add this to your bash setup by editing `.bashrc` or `.bash_profile`. Otherwise, just run this script to temporarily setup your environment.

```
source ./set_env.sh
```


## Download Embeddings
The PCA-based method requires pre-trained word embeddings. We trained our own embeddings with Wikipedia and PubMed data using [FastText](https://github.com/facebookresearch/fastText) and Gensim's implementation of [word2vec](https://radimrehurek.com/gensim/models/word2vec.html). You can download our embeddings (7.7GB compressed) using:

```
./download.sh
``` 

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
python benchmark.py --dataset cdr-supervised --model lstm --n_epochs 10 --debug
```

## Training Models

### Command Line Arguments
```
usage: benchmark.py [-h] [--info] [-d DATASET] [-m MODEL] [-c CONFIG]
                    [-g PARAM_GRID] [-p PARAMS] [-o OUTDIR]
                    [-N N_MODEL_SEARCH] [-E N_EPOCHS] [-W N_WORKERS]
                    [-H HOST_DEVICE] [--debug] [--seed SEED] [--quiet]

optional arguments:
  -h, --help            show this help message and exit
  --info                print all benchmark datasets
  -d DATASET, --dataset DATASET
                        dataset name
  -m MODEL, --model MODEL
                        model name
  -c CONFIG, --config CONFIG
                        load model config JSON
  -g PARAM_GRID, --param_grid PARAM_GRID
                        load manual parameter grid from JSON
  -p PARAMS, --params PARAMS
                        load `key=value,...` pairs from command line
  -o OUTDIR, --outdir OUTDIR
                        save model to outdir
  -N N_MODEL_SEARCH, --n_model_search N_MODEL_SEARCH
                        number of models to search over
  -E N_EPOCHS, --n_epochs N_EPOCHS
                        number of training epochs
  -W N_WORKERS, --n_workers N_WORKERS
                        number of grid search workers
  -H HOST_DEVICE, --host_device HOST_DEVICE
                        Host device (GPU|CPU)
  --debug               train on data subset
  --seed SEED           random model seed
  --quiet               suppress logging
```


### Train a basic BiLSTM

```
python benchmark.py --dataset cdr-supervised --model lstm -N 1 -E 50
```

This trains a basic BiLSTM *without* pre-trained embeddings for 50 epochs. 

### Train a BiLSTM + attention model with FastText embeddings

You can override any of the parameters specified in the model JSON by passing in a string of the format `<PARAM_NAME>=<VALUE>` (separating key/value pairs by a comma) to `-p`

```
python benchmark.py -d cdr-supervised -c configs/lstm.json -p attention=True,word_emb_dim=300,word_emb_path="data/embs/pubmed/fastText/300/pubmed.d300.w30.neg10.fasttext.vec" -N 1 -E 50
```

### Train a PCA + exp. decay model with FastText embeddings

```
python benchmark.py -d cdr-supervised -c configs/pca-kernel.json -p word_emb_dim=300,word_emb_path="data/embs/pubmed/fastText/300/pubmed.d300.w30.neg10.fasttext.vec" -N 1 -E 50
```

## Model Search
Our config files contain the same hyperparameter search space used in our experiments, just pass in a value >1 for `n_model_search`. 

```
python benchmark.py -d cdr-supervised -c configs/lstm.json -N 5 -E 50
```

We searched over 50 models for our inital experiments. The top 5 parameter configurations we found for BiLSTMs and PCA are in `configs/top5/`.


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
