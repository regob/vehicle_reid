<h1 align="center"> Vehicle ReID </h1>
<h2 align="center"> Strong, Small, Unfriendly </h2>

Baseline code for vehicle reID. Based on [layumi's person re-id repo](https://github.com/layumi/Person_reID_baseline_pytorch).

### Train
Train a model by e.g:

```bash
python3 reid/vehicle_reid/train.py --data_dir=datasets/ --name=resnet_ibn --train_csv_path=datasets/annot/id_split_verzal.csv --val_csv_path=datasets/annot/id_split_ver_val.csv --save_freq=1 --fp16 --ibn --contrast --erasing_p=0.5 --total_epoch=13
```

### Test and evaluate
Use trained model to extract feature by
```bash
python3 test.py --model_opts model/resnet_ibn_contrastive/opts.yaml --checkpoint model/resnet_ibn_contrastive/net_14.pth --query_csv_path ../../datasets/annot/id_split_cityflow_query.csv --gallery_csv_path ../../datasets/annot/id_split_cityflow_gallery.csv --data_dir ../../datasets/ --batchsize=8
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

### Zala test
Private test data with 100 ids.

| model | train data | Rank@1 | mAP | 
|---|---|---|---|
| Resnet50-ibn + contrastive | VeRi-Wild | 0.980| 0.706 |
| Resnet50-ibn + contrastive | VeRi-Wild + Zala | 0.994 | 0.904 |


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


