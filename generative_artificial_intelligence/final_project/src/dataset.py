"""FFHQ-64x64 dataset loader."""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class FFHQDataset(Dataset):
    def __init__(self, root, split="train", val_size=2000):
        self.root = root
        all_imgs = sorted([
            os.path.join(dp, f)
            for dp, _, fns in os.walk(root)
            for f in fns if f.endswith(".png")
        ])
        if split == "train":
            self.imgs = all_imgs[val_size:]
        else:
            self.imgs = all_imgs[:val_size]

        self.transform = T.Compose([
            T.ToTensor(),        # [0,1]
            T.Normalize([0.5] * 3, [1.0] * 3),  # [-0.5, 0.5]
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        return self.transform(img)


def get_loaders(data_root, batch_size=16, num_workers=16, val_size=2000):
    train_ds = FFHQDataset(data_root, split="train", val_size=val_size)
    val_ds = FFHQDataset(data_root, split="val", val_size=val_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)
    return train_loader, val_loader
