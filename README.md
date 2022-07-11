<h1 align="center"> Vehicle ReID </h1>
<h2 align="center"> Strong, Small, Unfriendly </h2>

Baseline code for vehicle reID. Based on [layumi's person re-id repo](https://github.com/layumi/Person_reID_baseline_pytorch).

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

The `train.py` training script supports many options for choosing a model and
loss. A ResNet50-ibn with contrastive loss is used as an example. Notable
parameters:
* `--fp16`: Use half precision training. The models and inputs are converted
  to fp16.
* `--name`: Name of directory to create for the model under `model/`.
* `--batchsize`: Batch size for training.
* `--total_epoch`: Number of epochs to run.
* `--lr`: Learning rate.
* `--save_freq`: How often to save checkpoints (number of epochs, default 1)
* `--checkpoint`: Checkpoint weights to load for the model (saved from
  previous run).
* `--start_epoch`: Start epoch (by default 0), if checkpoint loaded has to be
  consistent with it.

To train a Resnet50-ibn from

```bash
python3 train.py \
    --data_dir=datasets/ \
    --name=resnet_debug  \
    --train_csv_path=datasets/annot/id_split_train.csv \
    --val_csv_path=datasets/annot/id_split_val.csv \
    --save_freq=1 \
    --fp16 \
    --ibn \
    --contrast \
    --total_epoch=20
```

### Test and evaluate

Use trained model to extract features and evaluate metrics by:

```bash
python3 test.py \
    --model_opts model/resnet_ibn_contrastive/opts.yaml \
    --checkpoint model/resnet_ibn_contrastive/net_14.pth \
    --query_csv_path datasets/annot/id_split_cityflow_query.csv \
    --gallery_csv_path datasets/annot/id_split_cityflow_gallery.csv \
    --data_dir datasets/ \
    --batchsize=8
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.



## Results
### Cityflow 
The test data is created here from 200 ids from the cityflow labeled train data.
Models are trained on VeRi-Wild, VehicleX-SPGAN, and Zala datasets.

| model | train data | Rank@1 | mAP | 
|---|---|---|---|
| Resnet50-ibn + contrastive | VeRi-Wild | 0.998| 0.359 | 
| Resnet50-ibn + contrastive | VeRi-Wild + Zala | 0.995 | 0.408 |
| Resnet50-ibn + contrastive | VeRi-Wild + Zala + VehicleX | 0.998 |0.437 |

### Zala test
Private test data with 100 ids.

| model | train data | Rank@1 | mAP | 
|---|---|---|---|
| Resnet50-ibn + contrastive | VeRi-Wild | 0.980| 0.706 |
| Resnet50-ibn + contrastive | VeRi-Wild + Zala | 0.994 | 0.904 |
| Resnet50-ibn + contrastive | VeRi-Wild + Zala + VehicleX |0.994 |0.9032 |



## Citation
The following paper uses and reports the result of the baseline model. You may cite it in your paper.
```bib
@article{zheng2019joint,
  title={Joint discriminative and generative learning for person re-identification},
  author={Zheng, Zhedong and Yang, Xiaodong and Yu, Zhiding and Zheng, Liang and Yang, Yi and Kautz, Jan},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```


