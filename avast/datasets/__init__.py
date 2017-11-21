from __future__ import print_function
import os
import sys
from .base import CdrDataset, SpouseDataset


datasets = {}

# Relation Extraction (RE) tasks
datasets["cdr-supervised"]    = lambda name: CdrDataset(name, data_path="data/db/cdr.db")
datasets["cdr-dp"]            = lambda name: CdrDataset(name, data_path="data/db/cdr.db", lfs=True)
datasets["spouse-supervised"] = lambda name: SpouseDataset(name, data_path="data/db/spouse.db")
datasets["spouse-dp"]         = lambda name: SpouseDataset(name, data_path="data/db/spouse.db", lfs=True)


def subsample_dataset(dataset, percentage=0.10):
    """
    Create a subsample of the dataset

    :param dataset:
    :param percentage:
    :return:
    """
    i = int(percentage * float(len(dataset.X_train)))
    dataset.X_train, dataset.y_train = dataset.X_train[0:i], dataset.y_train[0:i]

    i = int(percentage * float(len(dataset.X_dev)))
    dataset.X_dev, dataset.y_dev = dataset.X_dev[0:i], dataset.y_dev[0:i]

    i = int(percentage * float(len(dataset.X_dev)))
    dataset.X_test, dataset.y_test = dataset.X_test[0:i], dataset.y_test[0:i]

    return dataset


def load_dataset(name, percentage=1.0):
    """
    Helper function to load named benchmark datasets

    :param name:
    :param subsample:   generate subset of all data (for debugging)
    :return:
    """
    if name not in datasets:
        sys.stderr.write('Fatal Error! Dataset {} not found!\n'.format(name))
    d = datasets[name](name)
    # sample subset for quick testing
    return subsample_dataset(d) if percentage < 1.0 else d

