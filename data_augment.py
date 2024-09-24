import random

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from torchvision import transforms


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            # (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class cons_dataset(Dataset):
    def __init__(self, fault_data, fault_label):
        self.fault_data = fault_data
        self.fault_label = fault_label
        self.data_enhance_fun = {
            '0': transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0),
                                                                  ratio=(0.75, 1.33)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])]),  # 随机裁剪
            '1': transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0),
                                                                  ratio=(0.75, 1.33)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])]),  # 随机裁剪和翻转
            '2': transforms.Compose([transforms.ToTensor(),
                                     Cutout(n_holes=1, length=200),
                                     transforms.Resize((224, 224)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])]),  # Cutout
            '3': transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                     transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])]),  # 高斯噪声
            '4': transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                     transforms.Grayscale(3),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])]),  # 灰度图像
            '5': transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])      # 颜色变化
        }

    def __len__(self):
        return len(self.fault_data)

    def __getitem__(self, item):
        # 制作label
        # label = np.asarray([0, 0, 0, 0])
        """label = []
        label[self.fault_label[item]] = 1"""
        label = self.fault_label[item]

        # 读取图片
        num1 = random.randint(0, 5)
        num2 = random.randint(0, 5)
        trans1 = self.data_enhance_fun[str(num1)]
        trans2 = self.data_enhance_fun[str(num2)]
        image = Image.open(self.fault_data[item][:-1])
        # image = Image.open(self.fault_data[item])
        if trans1:
            image_trans1 = trans1(image)

        if trans2:
            image_trans2 = trans2(image)

        sample = {'imidx': item, 'image_trans1': image_trans1, 'image_trans2': image_trans2, 'label': label}

        return sample
