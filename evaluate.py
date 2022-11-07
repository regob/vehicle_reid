import scipy.io
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Evaluate precomputed query and gallery features.")
parser.add_argument("--gpu", action="store_true", help="Use gpu")
parser.add_argument("--no_cams", action="store_true",
                    help="dont remove samples with same id and same cam as the query from the gallery.")
parser.add_argument("--K", type=int, default=-1,
                    help="If provided mAP@K will be calculated, else the same range will be used.")
args = parser.parse_args()

#######################################################################
# Evaluate

def evaluate(qf, ql, gf, gl, qc=None, gc=None, K=100):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    index = np.argsort(score)
    index = index[::-1]    
    # index has the scores decreasing at this point
    
    query_index = np.argwhere(gl == ql)
    junk_index = np.argwhere(gl < 0)

    # if camera labels are provided, exclude gallery images with the same camera and id as the query
    if qc is not None and gc is not None:
        camera_index = np.argwhere(qc == gc)
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        camera_junk = np.intersect1d(camera_index, query_index, assume_unique=True)
        junk_index = np.append(junk_index, camera_junk)
    else:
        good_index = query_index

    # only calc up to K
    if K < len(index):
        index = index[:K]
        good_index = np.intersect1d(index, good_index, assume_unique=True)
        junk_index = np.intersect1d(index, junk_index, assume_unique=True)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################
device = torch.device("cuda") if args.gpu else torch.device("cpu")

result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]
query_cam = result['query_cam'].reshape(-1)
gallery_cam = result['gallery_cam'].reshape(-1)
use_cam = (len(gallery_cam) > 0 and len(query_cam) > 0) and not args.no_cams

query_feature = query_feature.to(device)
gallery_feature = gallery_feature.to(device)

K = args.K if args.K >= 1 else len(gallery_label)
CMC = torch.IntTensor(K).zero_()
ap = 0.0

for i in range(len(query_label)):
    qc = query_cam[i] if use_cam else None
    gc = gallery_cam if use_cam else None
    ap_tmp, CMC_tmp = evaluate(
        query_feature[i], query_label[i], gallery_feature, gallery_label,
        qc, gc, K
    )
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' %
      (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
