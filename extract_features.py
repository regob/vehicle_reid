from __future__ import print_function, division

import argparse
import math
import time
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import scipy.io
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from load_model import load_model_from_opts
from dataset import ImageDataset
from tool.extract import extract_feature

torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument("--model_opts", required=True,
                    type=str, help="model saved options")
parser.add_argument("--checkpoint", required=True,
                    type=str, help="model checkpoint to load")
parser.add_argument("--csv_path", required=True,
                    type=str, help="csv to contain metadata for the images")
parser.add_argument("--data_dir", type=str, required=True,
                    help="root directory for image datasets")
parser.add_argument("--output_path", default="pytorch_result.mat",
                    help="file to write output features into.")
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--ms', default='1', type=str,
                    help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--num_workers', default=0, type=int)
opt = parser.parse_args()


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
interpolation = 3 if torchvision_version[0] == 0 and torchvision_version[1] < 13 else \
    transforms.InterpolationMode.BICUBIC

data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=interpolation),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


df = pd.read_csv(opt.csv_path)
classes = list(df["id"].unique())

image_dataset = ImageDataset(opt.data_dir, df, "id", classes, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,
                                         shuffle=False, num_workers=opt.num_workers)



######################################################################
# Load model
# ----------

print('------- Feature extraction -----------')
print("Running on: {}".format(device))

model = load_model_from_opts(opt.model_opts, ckpt=opt.checkpoint,
                             remove_classifier=True)
model.eval()
model.to(device)

# Extract features
since = time.time()
features, labels = extract_feature(model, dataloader, device, ms)

time_elapsed = time.time() - since
print('Complete in {:.0f}m {:.2f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

result = {'features': features.numpy(), 'labels': labels}
scipy.io.savemat(opt.output_path, result)

print("Feature extraction finished.")
