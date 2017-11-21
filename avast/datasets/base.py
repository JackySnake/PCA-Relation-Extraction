import os
import sys
import logging
import numpy as np


logger = logging.getLogger(__name__)

class Dataset(object):

    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def score(self,):
        raise NotImplementedError()

    def error_analysis(self):
        raise NotImplementedError()

    def print_summary(self, splits):
        """

        :param splits:
        :param name:
        :return:
        """
        logger.info("-" * 40)
        logger.info("[{}] Candidate Summary".format(self.name))
        logger.info("-" * 40)
        for n, candidates in splits.items():
            logger.info('[{0}]: {1} candidates'.format(n, len(candidates)))


class CdrDataset(Dataset):
    """
    Chemical-Disease Relation Extraction Task
    NOTES:
        - We assume our data is already cleaned and loaded into a sqlite3 database.
        - Weak supervision (via Snorkel labeling functions) have already been applied to generate marginals

    """
    def __init__(self, name, data_path, lfs=None, verbose=True):
        super(CdrDataset, self).__init__(name, verbose)
        assert os.path.exists(data_path)

        os.environ['SNORKELDB'] = "sqlite:///{}".format(data_path)
        logger.info("SQL connection {}".format(os.environ['SNORKELDB']))

        # Hack to prevent snorkel.db creation
        from snorkel import SnorkelSession
        from snorkel.models import candidate_subclass
        from snorkel.annotations import load_gold_labels, load_marginals

        self.session = SnorkelSession()
        self.class_type = candidate_subclass('ChemicalDisease', ['chemical', 'disease'])

        self.X_train  = self.session.query(self.class_type).filter(self.class_type.split == 0).all()
        self.X_dev    = self.session.query(self.class_type).filter(self.class_type.split == 1).all()
        self.X_test   = self.session.query(self.class_type).filter(self.class_type.split == 2).all()

        if self.verbose:
            splits = {"Train":self.X_train, "Dev":self.X_dev, "Test":self.X_test}
            self.print_summary(splits)
        if len(self.X_train) == 0:
            logger.error("Fatal error - no candidates found in database")
            sys.exit()

        self.y_train = load_marginals(self.session, split=0)

        self.y_gold_train = load_gold_labels(self.session, annotator_name='gold', split=0)
        self.y_gold_dev   = load_gold_labels(self.session, annotator_name='gold', split=1)
        self.y_gold_test  = load_gold_labels(self.session, annotator_name='gold', split=2)

        if not lfs:
            # convert to 0/1 marginals
            self.y_train = (np.ravel(self.y_gold_train.toarray()) + 1.) / 2
            self.y_dev   = (np.ravel(self.y_gold_dev.toarray()) + 1.) / 2
            self.y_test  = (np.ravel(self.y_gold_test.toarray()) + 1.) / 2

        else:
            self.y_train = self.y_train
            self.y_dev   = (np.ravel(self.y_gold_dev.toarray()) + 1.) / 2
            self.y_test  = (np.ravel(self.y_gold_test.toarray()) + 1.) / 2

    @property
    def data(self):
        return {
            'train':(self.X_train, self.y_train),
            'dev':  (self.X_dev, self.y_dev),
            'test': (self.X_test, self.y_test)
        }


class SpouseDataset(Dataset):
    """
    Spouse Relation Extraction Task

    NOTES:
        - We assume our data is already cleaned and loaded into a sqlite3 database.
        - Weak supervision (via Snorkel labeling functions) have already been applied to generate marginals
    """
    def __init__(self, name, data_path, lfs=None):
        super(SpouseDataset, self).__init__(name)

        os.environ['SNORKELDB'] = "sqlite:///{}".format(data_path)
        logger.info("SQL connection {}".format(os.environ['SNORKELDB']))

        # Hack to prevent snorkel.db creation
        from snorkel import SnorkelSession
        from snorkel.models import candidate_subclass
        from snorkel.annotations import load_gold_labels, load_marginals

        self.session = SnorkelSession()
        self.class_type = candidate_subclass('Spouse', ['person1', 'person2'])

        self.X_train  = self.session.query(self.class_type).filter(self.class_type.split == 0).all()
        self.X_dev    = self.session.query(self.class_type).filter(self.class_type.split == 1).all()
        self.X_test   = self.session.query(self.class_type).filter(self.class_type.split == 2).all()

        if self.verbose:
            splits = {"Train":self.X_train, "Dev":self.X_dev, "Test":self.X_test}
            self.print_summary(splits)

        self.y_train = load_marginals(self.session, split=0)

        self.y_gold_train = load_gold_labels(self.session, annotator_name='gold', split=0)
        self.y_gold_dev   = load_gold_labels(self.session, annotator_name='gold', split=1)
        self.y_gold_test  = load_gold_labels(self.session, annotator_name='gold', split=2)

        if not lfs:
            # convert to 0/1 marginals
            self.y_train = (np.ravel(self.y_gold_train.toarray()) + 1.) / 2
            self.y_dev   = (np.ravel(self.y_gold_dev.toarray()) + 1.) / 2
            self.y_test  = (np.ravel(self.y_gold_test.toarray()) + 1.) / 2

        else:
            self.y_train = self.y_train
            self.y_dev   = (np.ravel(self.y_gold_dev.toarray()) + 1.) / 2
            self.y_test  = (np.ravel(self.y_gold_test.toarray()) + 1.) / 2

    @property
    def data(self):
        return {
            'train':(self.X_train, self.y_train),
            'dev':  (self.X_dev, self.y_dev),
            'test': (self.X_test, self.y_test)
        }
