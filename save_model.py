import os
import sys
import torch
from torch import nn
from torchvision import transforms

import pandas as pd
import yaml
import argparse

from model import ft_net
from tools.dataset import ImageDataset

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

h, w = 224, 224
transform_val_list = [
    transforms.Resize(size=(h, w), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
val_transform = transforms.Compose(transform_val_list)

val_df = pd.read_csv(old_opts["val_csv_path"])
dataset = ImageDataset("", val_df, "id", transform=val_transform)
class_names = dataset.classes

return_feature = False
model = ft_net(len(class_names), old_opts["droprate"], old_opts["stride"], circle=return_feature,
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
