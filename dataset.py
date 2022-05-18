import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, normalize=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.normalize = normalize
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = os.path.join(self.image_dir, self.images[i])
        mask_path = os.path.join(self.mask_dir, self.images[i].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)/255

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.normalize:
            mean = image.mean((1, 2))
            std = image.std((1, 2))
            image = transforms.Normalize(mean=mean, std=std)(image)

        return image, mask


