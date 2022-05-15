import os
import torch
from torch import nn
import yaml
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_efficient, ft_net_NAS, PCB

sys.path.remove(os.path.dirname(SCRIPT_DIR))


def load_weights(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    if model.classifier.classifier[0].weight.shape != state["classifier.classifier.0.weight"].shape:
        state["classifier.classifier.0.weight"] = model.classifier.classifier[0].weight
        state["classifier.classifier.0.bias"] = model.classifier.classifier[0].bias
    model.load_state_dict(state)
    return model


def create_model(n_classes, kind="resnet", **kwargs):
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
    model = create_model(n_classes, kind, **kwargs)
    if ckpt:
        model = load_weights(model, ckpt)
    if remove_classifier:
        model.classifier.classifier = nn.Sequential()
        model.eval()
    return model


def load_model_from_opts(opts_file, ckpt=None, return_feature=False, remove_classifier=False):
    with open(opts_file, "r") as stream:
        opts = yaml.load(stream, Loader=yaml.FullLoader)
    n_classes = opts["nclasses"]
    droprate = opts["droprate"]
    stride = opts["stride"]
    linear_num = opts["linear_num"]
    model_subtype = "default" if "model_subtype" not in opts else opts["model_subtype"]

    if opts["use_dense"]:
        model = create_model(n_classes, "densenet", droprate=droprate, circle=return_feature,
                             linear_num=linear_num)
    elif opts["use_efficient"]:
        model=create_model(n_classes, "efficientnet", droprate=droprate,
                             circle=return_feature, linear_num=linear_num, model_subtype=model_subtype)
    elif opts["use_NAS"]:
        model=create_model(n_classes, "NAS", droprate=droprate,
                             linear_num=linear_num)
    elif opts["PCB"]:
        model=create_model(n_classes, "PCB")
    elif opts["use_hr"]:
        model = create_model(n_classes, "hr", droprate=droprate, circle=return_feature,
                             linear_num=linear_num)
    elif opts["use_swin"]:
        model = create_model(n_classes, "swin", droprate=droprate, stride=stride,
                             circle=return_feature, linear_num=linear_num)
    else:
        model = create_model(n_classes, "resnet", droprate=droprate, ibn=opts["ibn"],
                             stride=stride, circle=return_feature, linear_num=linear_num)

    if ckpt:
        load_weights(model, ckpt)
    if remove_classifier:
        model.classifier.classifier = nn.Sequential()
        model.eval()
    return model
