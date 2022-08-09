<h1 align="center"> Vehicle ReID </h1>
<h2 align="center"> Strong, Small, (Un)friendly </h2>

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/github/regob/vehicle_reid)](https://lgtm.com/projects/g/regob/vehicle_reid/context:python)
[![Total
alerts](https://img.shields.io/lgtm/alerts/github/regob/vehicle_reid?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/regob/vehicle_reid/)
[![Total LOC](https://img.shields.io/tokei/lines/github/regob/vehicle_reid)](https://img.shields.io/tokei/lines/github/regob/vehicle_reid?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


Baseline code for vehicle re-identification. Based on [layumi's person re-id
repo](https://github.com/layumi/Person_reID_baseline_pytorch).
The code is general, and works for any re-id task or dataset if the
required annotation files are created.

**A vehicle re-id tutorial is available now in a
[Kaggle notebook](https://www.kaggle.com/code/sosperec/vehicle-reid-tutorial/)
for training, evaluating, and using a model.**

## Installation
Tested on python3.8, but other versions could work too.
Clone the repo, then create a virtual environment:
```
$ python3 -m venv .venv
$ source .venv/bin/activate
```
And install the requirements (mostly torch and other utility packages):
```
$ pip3 install -r requirements.txt
```

## Datasets

| name | images | identities |
| :--- | :---: | :---: |
| VRIC | 60K | 5622 |
| Cityflow (v2) | 313K | 880 |
| VeRi-776 | 50K | 776 |
| VeRi-Wild | 416K | 40K |
| VehicleID | 221K | 26K |
| PKU-VD1 | 846K | 141K |
| PKU-VD2 | 807K | 79K |
| VehicleX | ∞ | ∞ (~170 models) |


For training a re-id model the dataset has to be prepared. The images can be stored in any structure, but a csv file is needed for the train and validation subsets. The csv files contain a row per image and have mandatory `path` and `id` columns (and possible other columns). E.g (brand, type, and color attributes are not available for CityFlow):
```
$ head id_split_cityflow_train.csv 
path,brand,type,color,id,subset
cityflow2_reid/image_train/006561.jpg,,,,54642,cityflow_train
cityflow2_reid/image_train/048961.jpg,,,,54520,cityflow_train
cityflow2_reid/image_train/017624.jpg,,,,54669,cityflow_train
```
The paths are relative to the **dataset root folder**, which is passed to the
scripts. An example directory structure could look like this:
```
|── datasets/
|    |── annot/
|        |── id_split_train.csv
|        |── id_split_val.csv
|    |── VeRi-Wild/
|        |── images/ 
│    ├── CityFlow/
|    |── VehicleX/ 
```

## Train

The `train.py` script can be used to train a model, and saves it into a subdirectory under `model`. The most important (mandatory) parameters:
- `--name`: Name of the model under the `model` subdirectory, it shouldn't exist yet, or things can be overwritten.
- `--data_dir`: The root directory of the datasets. 
- `--train_csv_path`: The csv containing training data.
- `--val_csv_path`: The csv containing validation data.

Other very important parameters:
- `--batchsize`: Batch size during training, should be reasonable (like 32, 64).
- `--model`: By default a Resnet50-ibn is trained. All options are: ['resnet', 'resnet_ibn', densenet', 'swin',
                    'NAS', 'hr', 'efficientnet']
- `--total_epoch`: Number of epochs - around 15 to 20 is needed at a minimum depending on the size of the dataset (with ~400 000 images I got decent results even after 10)
- `--warm_epoch`: Number of warmup epochs (increase learning rate gradually)
- `--save_freq`: Sets the frequency of saving a model in epochs (1: saving after each one, 2: after every second, etc), but the model is saved at the very end regardless.
- `--lr`: Learning rate.
- `--fp16`: Use Mixed precision training (convert to float16 automatically in
  forward pass)
- `--triplet`, `--contrast`, `--sphere`, `--circle`: Loss functions.

The following command is an example to train a Resnet50-ibn with contrastive loss:

```bash
python3 train.py \
    --data_dir=datasets/ \
    --name=resnet_debug  \
    --train_csv_path=datasets/annot/id_split_train.csv \
    --val_csv_path=datasets/annot/id_split_val.csv \
    --save_freq=1 \
    --fp16 \
    --contrast \
    --total_epoch=20
```

If we cannot complete the whole training in one session, it can be continued from a checkpoint by providing the following parameters:
- `--name`: it's value is the same as in the previous run
- `--checkpoint`: A model weight to be loaded, it is under the model's directory with the name of`net_X.pth`. (X = the number of epochs)
- `--start_epoch`: Epoch to continue from, if the checkpoint was `net_X.pth` this should be `X+1`.


## Test and evaluate

The `test.py` script computes embeddings for all gallery and query images and
evaluates various metrics. It can be run with the following parameters:
- `--data_dir`: Root dataset directory.
- `--query_csv_path`: Query annotation csv file.
- `--gallery_csv_path`: Gallery annot csv file.
- `--model_opts`: Path to the options used when training the model (e.g `~/vehicle_reid/model/resnet_debug/opts.yaml`)
- `--checkpoint`: Path to the checkpoint. The last one is always saved, but if we overtrained we can choose a previous one.
- `--batchsize`: Batch size for the model (does only affect performance, in
  case of low memory, this should be decreased).


Use trained model to extract features and evaluate metrics by:

```bash
python3 test.py \
    --data_dir=datasets/ \
    --query_csv_path=datasets/annot/id_split_cityflow_query.csv \
    --gallery_csv_path=datasets/annot/id_split_cityflow_gallery.csv \
    --model_opts=model/resnet_debug/opts.yaml \
    --checkpoint=model/resnet_debug/net_14.pth \
    --batchsize=8
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.

## Visualization

A simple script `visualize_test_queries.py` allows us to inspect and save some
queries. It can be run with the same parameters as `test.py`, but if we
already ran `test.py` on the current dataset, it saved the result as
`pytorch_result.mat`. The `--use_saved_mat` switch makes the visualization
script use this cached result instead of loading and executing the model:

```bash
python3 visualize_test_queries.py \
    --data_dir=datasets/ \
    --query_csv_path=datasets/annot/id_split_cityflow_query.csv \
    --gallery_csv_path=datasets/annot/id_split_cityflow_gallery.csv \
    --model_opts=model/resnet_debug/opts.yaml \
    --checkpoint=model/resnet_debug/net_14.pth \
    --use_saved_mat
```

A screenshot of the utility is below. The query image is the first with a
<span style="color:black">black</span> border, then come the gallery images
ordered descending by similarity to the query (so only the most similar images
are shown). The gallery images have a <span style="color:green">green</span> or <span
style="color:red">red</span> border depending on whether they are the same or
a different id than the query.
**The left and right arrows on the keyboard helps to navigate the queries.**

![Cityflow sample query.](assets/cityflow_sample_query.png)


## Results
### Cityflow 
The test data is created here from 200 ids from the cityflow labeled train data.
Models are trained on VeRi-Wild, VehicleX-SPGAN, and Zala datasets. So this is
a **cross-domain** experiment, no Cityflow data was used in training.

| model | train data | Rank@1 | mAP | 
|---|---|:---:|:---:|
| Resnet50-ibn + contrastive | VeRi-Wild | 0.998| 0.359 | 
| Resnet50-ibn + contrastive | VeRi-Wild + Zala | 0.995 | 0.408 |
| Resnet50-ibn + contrastive | VeRi-Wild + Zala + VehicleX | 0.998 |0.437 |

### VRIC

Models trained on random 75% of VRIC train, and tested on VRIC test.
All trainings used cross entropy as **id loss**, this is not listed in losses.
Params not in the table are constant: `--batchsize=64`, `--total_epoch=30`,
`--warm_epoch=3`, `--erasing_p=0.5`. No re-ranking was used.


| model           | metric losses | other params             | Rank@1    | Rank@5    | Rank@10   | mAP       |
|:----------------|:-------------:|:------------------------:|:---------:|:---------:|:---------:|:---------:|
| Resnet50-ibn    | `--contrast`  |                          | **0.549** | **0.804** | **0.873** | **0.607** |
| Resnet50-ibn    | `--sphere`    |                          | 0.552     | 0.795     | 0.862     | 0.606     |
| Resnet50-ibn    | `--instance`  |                          | 0.544     | 0.789     | 0.865     | 0.600     |
| Resnet50-ibn    | `--arcface`   |                          | 0.538     | 0.792     | 0.863     | 0.595     |
| Resnet50-ibn    | `--circle`    |                          | 0.514     | 0.782     | 0.857     | 0.574     |
| Resnet50-ibn    | `--contrast`  | `--label_smoothing=0.05` | 0.538     | 0.789     | 0.870     | 0.595     |
| Resnet50-ibn    |               |                          | 0.531     | 0.787     | 0.864     | 0.589     |
| Resnet50        |               |                          | 0.453     | 0.734     | 0.825     | 0.516     |
| Efficientnet-b0 |               |                          | 0.475     | 0.737     | 0.822     | 0.534     |



### Zala test
Private test data with 100 ids.

| model | train data | Rank@1 | mAP | 
|---|---|:---:|:---:|
| Resnet50-ibn + contrastive | VeRi-Wild | 0.980| 0.706 |
| Resnet50-ibn + contrastive | VeRi-Wild + Zala | 0.994 | 0.904 |
| Resnet50-ibn + contrastive | VeRi-Wild + Zala + VehicleX |0.994 |0.9032 |


## Citation
The following paper uses and reports the result of the original baseline model
(for person re-id). You may cite it in your paper.
```bib
@article{zheng2019joint,
  title={Joint discriminative and generative learning for person re-identification},
  author={Zheng, Zhedong and Yang, Xiaodong and Yu, Zhiding and Zheng, Liang and Yang, Yi and Kautz, Jan},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```


