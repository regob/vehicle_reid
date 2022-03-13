# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_efficient, ft_net_NAS, PCB, PCB_test

import pandas as pd
import tqdm
from tools.dataset import ImageDataset

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument("--model_path", required=True,
                    type=str, help="model to use")
parser.add_argument("--query_csv_path", required=True,
                    type=str, help="csv to contain query image data")
parser.add_argument("--gallery_csv_path", required=True,
                    type=str, help="csv to contain gallery image data")
parser.add_argument("--data_dir", type=str, required=True,
                    help="root directory for image datasets")
parser.add_argument('--name', default='ft_ResNet50',
                    type=str, help='model name')
parser.add_argument("--opts_yaml", type=str, default="",
                    help="yaml file with options saved from training")
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
# parser.add_argument('--multi', action='store_true', help='use multiple query')
# parser.add_argument('--fp16', action='store_true', help='use fp16.')
parser.add_argument('--ms', default='1', type=str,
                    help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
opt = parser.parse_args()

if opt.opts_yaml:
    with open(opt.opts_yaml, 'r') as stream:
        # for the new pyyaml via 'conda install pyyaml'
        config = yaml.load(stream, Loader=yaml.FullLoader)
else:
    config = {}

print('We use the scale: %s' % opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))


use_gpu = torch.cuda.is_available()
if not use_gpu:
    device = torch.device("cpu")
else:
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    device = torch.device("cuda")

######################################################################
# Load Data
# ---------
#
h, w = 224, 224

data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


query_df = pd.read_csv(opt.query_csv_path)
gallery_df = pd.read_csv(opt.gallery_csv_path)

image_datasets = {
    "query": ImageDataset(opt.data_dir, query_df, "id", transform=data_transforms),
    "gallery": ImageDataset(opt.data_dir, gallery_df, "id", transform=data_transforms)
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=2) for x in ['gallery', 'query']}


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloader):
    img_count = 0
    dummy = next(iter(dataloader))[0].to(device)
    output = model(dummy)
    feature_dim = output.shape[1]
    labels = []

    for idx, data in enumerate(tqdm.tqdm(dataloader)):
        X, y = data
        n, c, h, w = X.size()
        img_count += n
        ff = torch.FloatTensor(n, feature_dim).zero_().to(device)

        for lab in y:
            labels.append(lab)

        for i in range(2):
            if(i == 1):
                X = fliplr(X)
            input_X = Variable(X.to(device))
            for scale in ms:
                if scale != 1:
                    input_X = nn.functional.interpolate(
                        input_X, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_X)
                ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if idx == 0:
            features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])

        start = idx * opt.batchsize
        end = min((idx + 1) * opt.batchsize, len(dataloader.dataset))
        features[start:end, :] = ff
    return features, labels


######################################################################
# Load Collected data Trained model
print('-------test-----------')
print("Running on: {}".format(device))

model = torch.load(opt.model_path)
model.eval()
model.to(device)

# Extract feature
since = time.time()
with torch.no_grad():
    query_feature, query_labels = extract_feature(model, dataloaders['query'])
    gallery_feature, gallery_labels = extract_feature(
        model, dataloaders['gallery'])

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

# Save to Matlab for check
result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_labels,
          'query_f': query_feature.numpy(), 'query_label': query_labels}
scipy.io.savemat('pytorch_result.mat', result)

print("Feature extraction finished, starting evaluation ...")

result = os.path.join("model", opt.name, "result.txt")
os.system('python3 evaluate.py | tee -a %s' % (result))
