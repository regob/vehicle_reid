import os
import sys
import torch
from torch import nn
from torchvision import transforms

import yaml
import argparse

from model import ft_net

parser = argparse.ArgumentParser(
    description="Save model from training checkpoint")
parser.add_argument('--name', default='ft_ResNet50',
                    type=str, help='model name')
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Model checkpoint to load")
parser.add_argument("--output_path", type=str, required=True,
                    help="where to save the model")
opt = parser.parse_args()

# load original options from opts.yaml
model_dir = os.path.join("model", opt.name)
with open(os.path.join(model_dir, "opts.yaml"), "r") as stream:
    try:
        old_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)


return_feature = False
model = ft_net(old_opts["nclasses"], old_opts["droprate"], old_opts["stride"], circle=return_feature,
               ibn=old_opts["ibn"], linear_num=old_opts["linear_num"])


def load_network(network, path):
    if not os.path.isfile(path):
        path = os.path.join(model_dir, path)
    sdict = torch.load(path)
    network.load_state_dict(sdict)
    return network


model = load_network(model, opt.checkpoint)
model.classifier.classifier = nn.Sequential()
torch.save(model, opt.output_path)
