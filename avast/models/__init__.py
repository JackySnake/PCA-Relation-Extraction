from .params import *


def get_model_class(name):
    """
    Get model class object

    :param name:
    :return:
    """
    from snorkel.contrib.pca import PCA
    from snorkel.contrib.lstm import LSTM
    from snorkel.contrib.wclstm import WCLSTM
    assert name.lower() in ["lstm", "pca"]

    if name.lower() == "lstm":
        return LSTM
    elif name.lower() == "wclstm":
        return WCLSTM
    elif name.lower() == "pca":
        return PCA
    else:
        return None


def get_default_config(name):
    """
    Load default hyperparameters by type

    :param name:
    :return:
    """
    assert name.lower() in ["lstm","pca"]
    model = get_model_class(name)

    if name.lower() == "lstm":
        parameters = PyTorchLSTMParams()
        return model, parameters.model_class_params, parameters.model_hyperparams

    elif name.lower() == "pca":
        parameters = PyTorchPCAParams()
        return model, parameters.model_class_params, parameters.model_hyperparams
