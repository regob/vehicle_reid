from torch.utils.data import Dataset
from PIL import Image
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
