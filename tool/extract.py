import torch
from torch.autograd import Variable
import tqdm

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloader, device="cuda:0", ms=[1]):
    dummy = next(iter(dataloader))[0].to(device)
    output = model(dummy)
    feature_dim = output.shape[1]
    labels = []
    idx_start = 0

    for idx, data in enumerate(tqdm.tqdm(dataloader)):
        X, y = data
        n, c, h, w = X.size()
        ff = torch.FloatTensor(n, feature_dim).zero_().to(device)

        for lab in y:
            labels.append(lab)

        for i in range(2):
            if(i == 1):
                X = fliplr(X)
            input_X = Variable(X.to(device))
            for scale in ms:
                if scale != 1:
                    input_X = torch.nn.functional.interpolate(
                        input_X, scale_factor=scale, mode='bicubic', align_corners=False)
                with torch.no_grad():
                    outputs = model(input_X)
                ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if idx == 0:
            features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])

        idx_end = idx_start + n
        features[idx_start:idx_end, :] = ff
        idx_start = idx_end
    return features, labels

