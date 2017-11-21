from __future__ import print_function
import time
import logging
import numpy as np


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

