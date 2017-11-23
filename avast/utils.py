from __future__ import print_function
import sys
import time
import json
import logging
import numpy as np
from .models import *


logger = logging.getLogger(__name__)

def timeit(method):
    """
    Decorator function for timing

    :param method:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info('\n%r %2.2f sec' % (method.__name__, te-ts))
        return result
    return timed


def binary_scores_from_counts(ntp, nfp, ntn, nfn):
    """
    Precision, recall, and F1 scores from counts of TP, FP, TN, FN.
    Example usage:
        p, r, f1 = binary_scores_from_counts(*map(len, error_sets))
    """
    prec = ntp / float(ntp + nfp) if ntp + nfp > 0 else 0.0
    rec  = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def print_scores(ntp, nfp, ntn, nfn, title='Scores'):
    """
    Print classification metrics

    :param ntp:     num true positives
    :param nfp:     num false positives
    :param ntn:     num true negatives
    :param nfn:     num false negatives
    :param title:   table title
    :return:
    """
    prec, rec, f1 = binary_scores_from_counts(ntp, nfp, ntn, nfn)
    pos_acc = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    neg_acc = ntn / float(ntn + nfp) if ntn + nfp > 0 else 0.0
    logger.info("========================================")
    logger.info(title)
    logger.info("========================================")
    logger.info("Pos. class accuracy: {:.3}".format(pos_acc))
    logger.info("Neg. class accuracy: {:.3}".format(neg_acc))
    logger.info("Precision            {:.3}".format(prec))
    logger.info("Recall               {:.3}".format(rec))
    logger.info("F1                   {:.3}".format(f1))
    logger.info("----------------------------------------")
    logger.info("TP: {} | FP: {} | TN: {} | FN: {}".format(ntp, nfp, ntn, nfn))
    logger.info("========================================\n")


def error_analysis(X, y_true, y_pred):
    """

    :param X:
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = np.ravel(y_true).flatten()
    y_pred = np.ravel(y_pred).flatten()

    tp, fp, tn, fn = [], [], [], []
    for i,c in enumerate(X):
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += [c]
        elif y_pred[i] == 0 and y_true[i] == 0:
            tn += [c]
        elif y_pred[i] == 1 and y_true[i] == 1:
            tp += [c]
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += [c]
    return tp, fp, tn, fn


def print_parameter_space(gs, seed):

    logger.info("=" * 40)
    logger.info("Model Parameter Space (seed={}):".format(seed))
    logger.info("=" * 40)
    for i, params in enumerate(gs.search_space()):
        logger.info("{} {}".format(i, params))


def tune_beta_threshold(model, X_dev, y_dev, metric='f1'):

    best_score, best_b = 0, 0.0
    for b in range(1,10):
        b = 1.0 * b/10.0
        marginals = model.marginals(X_dev)
        y_pred = np.array([0 if marginals[i] < b else 1 for i in range(len(marginals))])
        tp, fp, tn, fn = error_analysis(X_dev, y_dev, y_pred)

        s = {}
        s['precision'], s['recall'], s['f1'] = binary_scores_from_counts(len(tp), len(fp), len(tn), len(fn))
        s['accuracy'] = 1.0 * (len(tp) + len(tn)) / (len(tp) + len(tn) + len(fp) + len(fn))

        if s[metric] >= best_score:
            best_score = s[metric]
            best_b = b

    logger.info("Tuned b={} ({}={}) on dev".format(best_b, metric, best_score))
    return best_b


def print_benchmark_summary():
    """
    Print datasets available for benchmarks

    :return:
    """
    from avast.datasets import datasets

    logger.info("=" * 40)
    logger.info("Datasets")
    logger.info("=" * 40)
    for name in sorted(datasets):
        logger.info("[+] {}".format(name))
    logger.info("-" * 40)


def print_key_pairs(v, title="Parameters"):
    """
    Print python dictionary key/value pairs

    :param v:       python dictionary
    :param title:   table title
    :return:
    """
    items = v.items() if type(v) is dict else v
    logger.info("-" * 40)
    logger.info(title)
    logger.info("-" * 40)
    for key,value in items:
        logger.info("{:<20}: {:<10}".format(key, value))
    logger.info("-" * 40)


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


def use_pretrained_embs(model_hyperparams, model_param_grid, manual_param_grid):
    """
    This ugly function checks if word embeddings are defined
    anywhere in our model configurations.

    :param model_hyperparams:
    :param model_param_grid:
    :param manual_param_grid:
    :return:
    """
    if 'load_emb' in model_hyperparams and model_hyperparams['load_emb']:
        return True
    if "word_emb_path" in model_hyperparams and model_hyperparams["word_emb_path"]:
        return True
    if "word_emb_path" in model_param_grid and model_param_grid["word_emb_path"]:
        return True
    if "word_emb_path" in manual_param_grid and manual_param_grid["param_names"]:
        return True
    return False


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
            # also override in the param grid
            if name in model_param_grid:
                model_param_grid[name] = [params[name]]

    logger.info(model_param_grid)
    # override model params from command line
    model_class_params['seed']       = args.seed
    model_class_params['n_threads']  = args.n_procs
    model_hyperparams['n_epochs']    = args.n_epochs
    model_hyperparams['host_device'] = args.host_device

    if verbose:
        print_key_pairs(model_hyperparams, "Model Hyperparameters")

    return model, model_class_params, model_hyperparams, model_param_grid, manual_param_grid