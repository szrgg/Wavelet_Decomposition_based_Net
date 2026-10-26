import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from scipy.ndimage import shift as nd_shift



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_flip(image, label):
    # 只进行左右翻转
    image = np.flip(image, axis=1).copy()
    label = np.flip(label, axis=1).copy()

    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):


    def __init__(self, output_size, augment=True):
        self.output_size = output_size
        self.augment = augment

    def __call__(self, sample):
        image, label = sample['image'], sample['label']


        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        if self.augment:

            if random.random() > 0.5:
                image, label = random_flip(image, label)
            if random.random() < 0.2:
                image, label = random_rotate(image, label)  # Integer angle: [-20, 19].
            if random.random() < 0.2:
                row_shift = random.randint(-10, 10)
                col_shift = random.randint(-10, 10)
                image = nd_shift(
                    image, shift=(row_shift, col_shift, 0), order=1, mode='reflect'
                )
                label = nd_shift(
                    label, shift=(row_shift, col_shift), order=0, mode='reflect'
                )

        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)


        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # [C, H, W]
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}



        return sample



# 自定义数据集类
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name + '.npz')
        data = np.load(data_path)
        image, label = data['image'], data['label']



        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = slice_name
        return sample
