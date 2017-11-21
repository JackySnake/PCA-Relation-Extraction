"""
Parameter definitions are strongly tied to the Snorkel grid search interface,
which requires 3 parameter sets to instantiate and search over models

    model_class_params: required to instantiate model class
    model_hyperparams:  model hyperparameter defaults
    model_param_grid:   search grid (NOTE: these override model defaults)

"""

class BaseParams(object):
    """
    All testable models share these hyperparameters
    """
    def __init__(self, seed=123, n_procs=1, print_freq=5, dev_ckpt_delay=0.1, patience=50):
        """
        :param seed:
        :param n_procs:
        :param print_freq: print updates and checkpoint models every N epochs
        :param dev_ckpt_delay: (Snorkel grid search) percentage of
                                epochs to delay model snapshots
        :param patience: (Snorkel grid search) number of epochs to continue
                         training without improvement of the dev set
        """
        self.model_class_params = {
            'seed':           seed,
            'n_threads':        n_procs,
        }
        self.model_hyperparams  = {}
        self.model_param_grid   = {}


class PyTorchLSTMParams(BaseParams):

    def __init__(self, model_hyperparams={}, seed=123, n_procs=1, print_freq=5,
                 dev_ckpt_delay=0.1, patience=50):
        """

        :param model_hyperparams:
        :param seed:
        :param n_procs:
        :param print_freq:
        :param dev_ckpt_delay:
        :param patience:
        """
        super(PyTorchLSTMParams, self).__init__(seed, n_procs, print_freq,
                                                dev_ckpt_delay, patience )

        self.model_hyperparams = {
            'print_freq':          print_freq,
            'dev_ckpt_delay':      dev_ckpt_delay,
            'patience':            patience,
            'batch_size':          128,
            'word_emb_dim':        300,
            'attention':           False,
            'load_emb':            False,
            'max_sentence_length': 100,
            'n_epochs':            200,
            'rebalance':           0.0
        }

        # override any hyperparameter found in model_hyperparams
        for param,value in model_hyperparams.items():
            self.model_hyperparams[param] = value


class PyTorchPCAParams(BaseParams):

    def __init__(self, model_hyperparams={}, seed=123, n_procs=1, print_freq=5,
                 dev_ckpt_delay=0.1, patience=50):
        """

        :param model_hyperparams:
        :param seed:
        :param n_procs:
        :param print_freq:
        :param dev_ckpt_delay:
        :param patience:
        """
        super(PyTorchLSTMParams, self).__init__(seed, n_procs, print_freq,
                                                dev_ckpt_delay, patience )

        self.model_hyperparams = {
            "print_freq"     : 5,
            "lr"             : 0.001,
            "word_emb_dim"   : 300,
            "dev_ckpt_delay" : 0.1,
            'patience'       : patience,
            "n_epochs"       : 200,
            "l"              : 0,
            "r"              : 1,
            "batch_size"     : 100,
            "window_size"    : 1,
            "asymmetric"     : False,
            "char"           : False,
            "cont_feat"      : True,
            "sent_feat"      : True
        }

        # override any hyperparameter found in model_hyperparams
        for param,value in model_hyperparams.items():
            self.model_hyperparams[param] = value



