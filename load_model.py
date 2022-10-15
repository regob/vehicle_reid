import os
import torch
from torch import nn
import yaml
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_efficient, ft_net_NAS, PCB

sys.path.remove(SCRIPT_DIR)


def load_weights(model, ckpt_path):
    """Loads weights of the model from a checkpoint file

    Paremeters
    ----------
    model: torch.nn.Module
        Model to load weights of (needs to have a model.classifier head).
    ckpt_path: str
        Path to the checkpoint file to load (e.g net_X.pth).

    Returns
    -------
    model: torch.nn.Module
        The model object with the loaded weights.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if model.classifier.classifier[0].weight.shape != state["classifier.classifier.0.weight"].shape:
        state["classifier.classifier.0.weight"] = model.classifier.classifier[0].weight
        state["classifier.classifier.0.bias"] = model.classifier.classifier[0].bias
    model.load_state_dict(state)
    return model


def create_model(n_classes, kind="resnet", **kwargs):
    """Creates a model of a given kind and number of classes"""
    if kind == "resnet":
        return ft_net(n_classes, **kwargs)
    elif kind == "densenet":
        return ft_net_dense(n_classes, **kwargs)
    elif kind == "hr":
        return ft_net_hr(n_classes, **kwargs)
    elif kind == "efficientnet":
        return ft_net_efficient(n_classes, **kwargs)
    elif kind == "NAS":
        return ft_net_NAS(n_classes, **kwargs)
    elif kind == "swin":
        return ft_net_swin(n_classes, **kwargs)
    elif kind == "PCB":
        return PCB(n_classes)
    else:
        raise ValueError("Model type cannot be created: {}".format(kind))


def load_model(n_classes, kind="resnet", ckpt=None, remove_classifier=False, **kwargs):
    """Loads a model of a given type and number of classes.

    Parameters
    ----------
    n_classes: int
        Number of classes at the head.
    kind: str
        Type of the model ('resnet', 'efficientnet', 'densenet', 'hr', 'swin', 'NAS', 'PCB').
    ckpt: Union[str, None]
        Path to the checkpoint to load or None.
    remove_classifier: bool
        Whether or not to remove the classifier head.
    **kwargs: params to pass to the model

    Returns
    -------
    model: torch.nn.Module
    """

    model = create_model(n_classes, kind, **kwargs)
    if ckpt:
        model = load_weights(model, ckpt)
    if remove_classifier:
        model.classifier.classifier = nn.Sequential()
        model.eval()
    return model


def load_model_from_opts(opts_file, ckpt=None, return_feature=False, remove_classifier=False):
    """Loads a saved model by reading its opts.yaml file.

    Parameters
    ----------
    opts_file: str
        Path to the saved opts.yaml file of the model
    ckpt: str
        Path to the saved checkpoint of the model (net_X.pth)
    return_feature: bool
        Shows whether the model has to return the feature along with the result in the forward
        function. This is needed for certain loss functions (circle loss).
    remove_classifier: bool
        Whether we have to remove the classifier block from the model, which is needed for
        training but not for evaluation

    Returns
    -------
    model: torch.nn.Module
        The model requested to be loaded.
    """

    with open(opts_file, "r") as stream:
        opts = yaml.load(stream, Loader=yaml.FullLoader)
    n_classes = opts["nclasses"]
    droprate = opts["droprate"]
    stride = opts["stride"]
    linear_num = opts["linear_num"]
    
    model_subtype = opts.get("model_subtype", "default")
    model_type = opts.get("model", "resnet_ibn")
    mixstyle = opts.get("mixstyle", False)

    if model_type in ("resnet", "resnet_ibn"):
        model = create_model(n_classes, "resnet", droprate=droprate, ibn=(model_type == "resnet_ibn"),
                             stride=stride, circle=return_feature, linear_num=linear_num,
                             model_subtype=model_subtype, mixstyle=mixstyle)
    elif model_type == "densenet":
        model = create_model(n_classes, "densenet", droprate=droprate, circle=return_feature,
                             linear_num=linear_num)
    elif model_type == "efficientnet":
        model = create_model(n_classes, "efficientnet", droprate=droprate,
                             circle=return_feature, linear_num=linear_num, model_subtype=model_subtype)
    elif model_type == "NAS":
        model = create_model(n_classes, "NAS", droprate=droprate,
                             linear_num=linear_num)
    elif model_type == "PCB":
        model = create_model(n_classes, "PCB")
    elif model_type == "hr":
        model = create_model(n_classes, "hr", droprate=droprate, circle=return_feature,
                             linear_num=linear_num)
    elif model_type == "swin":
        model = create_model(n_classes, "swin", droprate=droprate, stride=stride,
                             circle=return_feature, linear_num=linear_num)
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    if ckpt:
        load_weights(model, ckpt)
    if remove_classifier:
        model.classifier.classifier = nn.Sequential()
        model.eval()
    return model
