"""
Evaluation Pipeline

This is a lightweight script for testing models on various Snorkel datasets

"""
from __future__ import print_function
import os
import sys
import json
import torch
import pandas
import logging
import argparse
from avast.utils import *
from avast.models import *
from avast.datasets import load_dataset


# set reasonable pandas dataframe display defaults
pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

logger = logging.getLogger(__name__)

@timeit
def train(X_train, y_train, X_dev, y_dev, X_test, y_test, model_save_dir, model_class,
          model_class_params, model_hyperparams, model_param_grid, manual_param_grid=None,
          n_epochs=1, num_model_search=5, num_workers=1, seed=123, verbose=True):
    """
    Find the best model using random grid search OR train a single model
    based on the parameters set in `model_hyperparams`

    :param X_train:             training candidates
    :param y_train:             training labels (or marginals)
    :param X_dev:               development candidates
    :param y_dev:               development labels
    :param X_test:              test candidates
    :param y_test:              test labels
    :param model_save_dir:      save checkpoints and best models here
    :param model_class:         disc. model class (e.g., LSTM)
    :param model_class_params:  disc. model default class init params
    :param model_hyperparams:   disc. model defaults
    :param model_param_grid:    grid search space
    :param n_epochs:            training epochs (overridden by `model_param_grid`)
    :param num_model_search:    number of models to search over
    :param num_workers:         grid search workers
    :param seed:                PyTorch random seed
    :param verbose:             print all output
    :return:
    """
    from snorkel.learning import RandomSearch

    # Initialize pre-trained embedding dictionary
    if 'load_emb' in model_hyperparams and model_hyperparams['load_emb']:
        model_hyperparams["init_pretrained"] = {"train": X_train, "test": [X_dev, X_test]}

    if num_model_search > 1:
        gs = RandomSearch(model_class, model_param_grid, X_train, y_train,
                          n=num_model_search, seed=seed,
                          save_dir=model_save_dir,
                          model_class_params=model_class_params,
                          model_hyperparams=model_hyperparams,
                          manual_param_grid=manual_param_grid)

        print_parameter_space(gs, seed)
        model, run_stats = gs.fit(X_valid=X_dev, Y_valid=y_dev, n_threads=num_workers)

        # print random search results
        if verbose:
            logger.info(run_stats)
            p, r, f1 = model.score(X_dev, y_dev)
            logger.info("Dev Score: {:2.2f} / {:2.2f} / {:2.2f}".format(p*100, r*100, f1*100))
            logger.info(model_save_dir)

    else:
        model = model_class(**model_class_params)
        model.train(X_train, y_train, X_dev=X_dev, Y_dev=y_dev,
                    save_dir=model_save_dir, **model_hyperparams)

    return model


def score(X_test, y_test, model, b=0.5):
    """
    Compute scores on test set using the provided model

    :param X_test:
    :param y_test:
    :param model:
    :param b:
    :return:
    """
    marginals      = model.marginals(X_test)
    y_pred         = np.array([0 if marginals[i] < b else 1 for i in range(len(marginals))])
    tp, fp, tn, fn = error_analysis(X_test, y_test, y_pred)
    print_scores(len(tp), len(fp), len(tn), len(fn), title='Test Set Scores')


def convert_param_string(s):
    """
    Convert string of hyperparamters into typed dictionary
    e.g., `lr=0.001,rebalance=False,attention=True`

    :param s:
    :return:
    """
    config = dict([p.split("=") for p in s.split(",")])

    # force typecasting in this order
    types = [int, float]
    for param in config:
        v = config[param]
        for t in types:
            try:
                v = t(v)
            except:
                continue
            config[param] = v
            break
        if config[param] in ['true','True']:
            config[param] = True
        elif config[param] in ['false','False']:
            config[param] = False

    return config


def get_model_config(args, verbose=True):
    """
    Setup model configuration, including model class and all hyperparameter defaults.
    Model configs also include a default grid search space.

    :param model_name:
    :param manual_config:
    :param verbose:
    :return:
    """
    if args.config:
        args.config = json.load(open(args.config,"rU"))
        model = get_model_class(args.config[u"model"])
        model_class_params = args.config[u'model_class_params']
        model_hyperparams  = args.config[u'model_hyperparams']
        model_param_grid   = args.config[u'model_param_grid']
        logger.info("Loaded model config from JSON file...")

    # use model defaults
    elif args.model:
        model, model_class_params, model_hyperparams = get_default_config(args.model)
        model_param_grid = {}
        logger.info("Loaded model defaults...")

    else:
        logger.error("Please specify model config or model class type")
        sys.exit()

    # override parameter grid and model search num
    if args.param_grid:
        manual_param_grid = json.load(open(args.param_grid, "rU"))
        args.n_model_search = len(manual_param_grid[u'params'])
        logger.info("Using manual parameter grid, setting n_model_search={}".format(args.n_model_search))
    else:
        manual_param_grid = {}

    # custom model parameters
    if args.params:
        params = convert_param_string(args.params)
        # override any grid search settings
        logger.info("Overriding some model hyperparameters")
        # override any model_hyperparameter defaults
        for name in params:
            model_hyperparams[name] = params[name]

    # override model params from command line
    model_class_params['seed']       = args.seed
    model_class_params['n_threads']  = args.n_procs
    model_hyperparams['n_epochs']    = args.n_epochs
    model_hyperparams['host_device'] = args.host_device

    if verbose:
        print_key_pairs(model_hyperparams, "Model Hyperparameters")

    return model, model_class_params, model_hyperparams, model_param_grid, manual_param_grid


def main(args):

    # ------------------------------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------------------------------
    dataset = load_dataset(args.dataset, percentage=(0.10 if args.debug else 1.0))

    X_train, y_train = dataset.data['train']
    X_dev, y_dev     = dataset.data['dev']
    X_test, y_test   = dataset.data['test']

    # ------------------------------------------------------------------------------
    # Load Model and Hyperparameters
    # ------------------------------------------------------------------------------
    # Snorkel model search requires 3 parameter dictionaries
    # - model_class_params: required to initialize model
    # - model_hyperparams:  default hyperparameters
    # - model_param_grid:  hyperparameter search space
    # - manual_param_grid:  manually defined parameter configs (useful for loading known good configurations)

    # IMPORTANT: this needs to happen *after* the dataset is loaded in order to initialize Snorkel correctly
    model_class, model_class_params, model_hyperparams, model_param_grid, manual_param_grid = get_model_config(args)

    # ------------------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------------------
    # save models in this dir
    ts = int(time.time())
    model_save_dir = "checkpoints/{}_{}".format(args.dataset, ts)

    # Use grid search to select best model over `model_param_grid` parameters
    model = train(X_train, y_train, X_dev, y_dev, X_test, y_test, model_save_dir,
                  model_class, model_class_params, model_hyperparams, model_param_grid,
                  manual_param_grid=manual_param_grid,
                  n_epochs=args.n_epochs, num_model_search=args.n_model_search,
                  num_workers=args.n_workers, seed=args.seed, verbose=True)

    # ------------------------------------------------------------------------------
    # Score and (Optionally) Save Best Model
    # ------------------------------------------------------------------------------
    score(X_test, y_test, model, b=0.5)

    # save model
    if args.outdir:
        model_params = (args.dataset, args.model, args.n_epochs, args.n_model_search, ts, args.seed)
        model.save("best.{}.m{}.epochs{}.gs{}.ts{}.seed{}.model".format(*model_params), save_dir=model_save_dir)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--info", action="store_true", help="print all benchmark datasets")
    argparser.add_argument("-d", "--dataset", type=str, default="cdr-supervised", help="dataset name")
    argparser.add_argument("-m", "--model", type=str, default="lstm", help="model name")
    argparser.add_argument("-c", "--config", type=str, default=None, help="load model config JSON")
    argparser.add_argument("-g", "--param_grid", type=str, default=None, help="load manual parameter grid from JSON")
    argparser.add_argument("-p", "--params", type=str, default=None, help="load `key=value,...` pairs from command line")
    argparser.add_argument("-o", "--outdir", type=str, default=None, help="save model to outdir")

    argparser.add_argument("-N", "--n_model_search", type=int, default=1, help="number of models to search over")
    argparser.add_argument("-E", "--n_epochs", type=int, default=1, help="number of training epochs")
    argparser.add_argument("-M", "--n_procs", type=int, default=1, help="number processes (per model, CPU only)")
    argparser.add_argument("-W", "--n_workers", type=int, default=1, help="number of grid search workers")
    argparser.add_argument("-H", "--host_device", type=str, default="cpu", help="Host device (GPU|CPU)")

    argparser.add_argument("--debug", action="store_true", default=False, help="train on data subset")
    argparser.add_argument("--seed", type=int, default=123, help="random model seed")
    argparser.add_argument("--quiet", action="store_true", help="suppress logging")
    args = argparser.parse_args()

    # enable logging
    if not args.quiet:
        FORMAT = '%(levelname)s|%(name)s|  %(message)s'
        logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

    # dump a list of models and datasets
    if args.info:
        print_benchmark_summary()
        sys.exit()

    # check for CUDA support
    if not torch.cuda.is_available() and args.host_device.lower() == 'gpu':
        logger.error("Warning! CUDA not available, defaulting to CPU")
        args.host_device = "cpu"

    # print all argument variables
    print_key_pairs(args.__dict__.items(), title="Command Line Args")

    main(args)