from torch.utils.data import Dataset
from PIL import Image
import random
import os


class ImageDataset(Dataset):
    """A labeled image dataset whose annotation is provided as a DataFrame."""

    def __init__(self, img_root, df, target_label, classes="infer", transform=None, target_transform=None):
        self.img_root = img_root
        self.df = df
        self.target_label = target_label
        self.transform = transform
        self.target_transform = target_transform
        if classes == "infer":
            self.classes = list(set(df[target_label].values))
        else:
            self.classes = classes
            class_set = set(df[target_label].values)
            assert class_set.issubset(
                set(classes)), "Error: Classes in df are not a subset of provided classes."
        self.class_idx = {cl: idx for idx, cl in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def get_image(self, idx):
        """Returns the image at a given index of the dataset."""
        row = self.df.loc[idx]
        pth = os.path.join(self.img_root, row["path"])
        image = Image.open(pth).convert("RGB")
        return image

    def __getitem__(self, idx):
        """Returns the transformed image and label at a given index."""
        row = self.df.loc[idx]
        pth = os.path.join(self.img_root, row["path"])
        image = Image.open(pth).convert("RGB")
        label = self.class_idx[row[self.target_label]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class BatchSampler:
    def __init__(self, dataset, batch_size, samples_per_class, drop_last=True):
        """Samples a dataset into batches, with the given number of samples per class if possible."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last
        self.batches, self.batch_idx = [], 0
        

    def __iter__(self):
        ids = self.dataset.df[self.dataset.target_label]
        ids = ids.sample(frac=1.0)
        samples_for_id = {}
        for idx, cls in ids.items():
            samples_for_id.setdefault(cls, []).append(idx)

        # create patches of size: samples_per_class
        patches = []
        for _, samples in samples_for_id.items():
            for i in range(0, len(samples), self.samples_per_class):
                patches.append(samples[i:i + self.samples_per_class])

        random.shuffle(patches)
        self.batches, self.batch_idx = [[]], 0
        for patch in patches:
            last_batch = self.batches[-1]
            if len(patch) + len(last_batch) <= self.batch_size:
                last_batch.extend(patch)
            else:
                num_needed = self.batch_size - len(last_batch)
                last_batch.extend(patch[:num_needed])
                self.batches.append(patch[num_needed:])
        if len(self.batches[-1]) < self.batch_size:
            self.batches.pop()
        return self

    def __len__(self):
        n_samples = len(self.dataset.df)
        n_batches = n_samples // self.batch_size
        if not self.drop_last and n_samples % self.batch_size != 0:
            n_batches += 1
        return n_batches

    def __next__(self):
        self.batch_idx += 1
        if self.batch_idx > len(self.batches):
            raise StopIteration()
        return self.batches[self.batch_idx - 1]

    
    
