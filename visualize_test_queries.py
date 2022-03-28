import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms
import argparse
import os
import sys
import matplotlib.pyplot as plt
import tqdm
import math
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from load_model import load_model_from_opts
from tools.dataset import ImageDataset

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(
    description="Show sample queries and retrieved gallery images for a reid model")
parser.add_argument("--model_opts", required=True,
                    type=str, help="model to use")
parser.add_argument("--checkpoint", required=True,
                    type=str, help="checkpoint to load for model.")
parser.add_argument("--query_csv_path", default="../../datasets/id_split_cityflow_query.csv",
                    type=str, help="csv to contain query image data")
parser.add_argument("--gallery_csv_path", default="../../datasets/id_split_cityflow_gallery.csv",
                    type=str, help="csv to contain gallery image data")
parser.add_argument("--data_dir", type=str, default="../../datasets/",
                    help="root directory for image datasets")
parser.add_argument("--input_size", type=int, default=224,
                    help="Image input size for the model")
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--num_images", type=int, default=29,
                    help="number of gallery images to show")
parser.add_argument("--imgs_per_row", type=int, default=6)
args = parser.parse_args()

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

h, w = args.input_size, args.input_size


######################################################################
# Load Data
# ---------
#

data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


query_df = pd.read_csv(args.query_csv_path)
gallery_df = pd.read_csv(args.gallery_csv_path)
classes = list(pd.concat([query_df["id"], gallery_df["id"]]).unique())

#gallery_df = gallery_df.loc[:640].copy()


image_datasets = {
    "query": ImageDataset(args.data_dir, query_df, "id", classes, transform=data_transforms),
    "gallery": ImageDataset(args.data_dir, gallery_df, "id", classes, transform=data_transforms),
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchsize,
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
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_features(model, dataloader):
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
            outputs = model(input_X)
            ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if idx == 0:
            features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])

        start = idx * args.batchsize
        end = min((idx + 1) * args.batchsize, len(dataloader.dataset))
        features[start:end, :] = ff
    return features, labels


def extract_feature(model, img):
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
    img = img.to(device)
    feature = model(img).reshape(-1)

    img = fliplr(img)
    flipped_feature = model(img).reshape(-1)
    feature += flipped_feature

    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)


def get_scores(query_feature, gallery_features):
    query = query_feature.view(-1, 1)
    score = torch.mm(gallery_features, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score


def show_query_result(axes, query_img, gallery_imgs, query_label, gallery_labels):
    query_trans = transforms.Pad(4, 0)
    good_trans = transforms.Pad(4, (0, 255, 0))
    bad_trans = transforms.Pad(4, (255, 0, 0))

    for idx, img in enumerate([query_img] + gallery_imgs):
        img = img.resize((128, 128))
        if idx == 0:
            img = query_trans(img)
        elif query_label == gallery_labels[idx - 1]:
            img = good_trans(img)
        else:
            img = bad_trans(img)

        ax = axes.flat[idx]
        ax.imshow(img)

    for i in range(len(axes.flat)):
        ax = axes.flat[i]
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        ax.axis("off")


######################################################################
# Run queries
# -----------
#


model = load_model_from_opts(
    args.model_opts, args.checkpoint, remove_classifier=True)
model.eval()
model.to(device)

print("Computing gallery features ...")

with torch.no_grad():
    gallery_features, gallery_labels = extract_features(
        model, dataloaders["gallery"])

dataset = image_datasets["query"]
queries = list(range(len(dataset)))
random.shuffle(queries)


def on_key(event):
    global curr_idx
    if event.key == "left":
        curr_idx = (curr_idx - 1) if curr_idx > 0 else len(queries) - 1
    elif event.key == "right":
        curr_idx = (curr_idx + 1) if curr_idx < len(queries) - 1 else 0
    else:
        return
    refresh_plot()


def refresh_plot():
    X, y = dataset[curr_idx]
    with torch.no_grad():
        q_feature = extract_feature(model, X).cpu()

    gallery_scores = get_scores(q_feature, gallery_features)
    idx = np.argsort(gallery_scores)[::-1]
    g_labels = np.array(gallery_labels)[idx]

    q_img = dataset.get_image(curr_idx)
    g_imgs = [image_datasets["gallery"].get_image(i)
              for i in idx[:args.num_images]]
    show_query_result(axes, q_img, g_imgs, y, g_labels)
    fig.canvas.draw()
    fig.canvas.flush_events()


n_rows = math.ceil((1 + args.num_images) / args.imgs_per_row)
fig, axes = plt.subplots(n_rows, args.imgs_per_row, figsize=(12, 15))
fig.canvas.mpl_connect('key_press_event', on_key)

curr_idx = 0
refresh_plot()
plt.show()
