import os
import cv2
import enum
import string
import pathlib

import numpy as np

from pathlib import Path
from typing import List
from collections import namedtuple
from torch.utils.data import Dataset


image_info = namedtuple(
    'data_info_tuple',
    'name, image, label'
)


class DatasetType(enum.Enum):
    TRAIN = 'training_dataset'
    VALIDATION = 'validation_dataset'
    TEST = 'test_dataset'

# TODO: Napraviti skriptu (ne kao Petar) koja će provjeriti sadrži li maska bijeli piksel i staviti joj label
class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir: string,
        img_dir: string,
        type: DatasetType = DatasetType.TRAIN,
        patch_size: int = 128,
        transform=None
    ) -> None:
        self.patch_size = patch_size
        self.transform = transform
        self.images_data = self.preload_image_data(data_dir, img_dir, type)

    def preload_image_data_dir(
        self,
        data_dir: string,
        img_dir: string,
        type: DatasetType
    ):
        dataset_files: List = []
        path = pathlib.Path(data_dir, f'{type.value}.txt')

        with open(path, mode='r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                path = pathlib.Path(data_dir, img_dir, line.strip())
                mask = cv2.imread(pathlib.Path(path, 'Mask'))[:, :, ::-1]
                print(np.unique(mask))

                data_info = image_info(
                    line.strip(),
                    pathlib.Path(path, 'Image'),
                    1
                )
                dataset_files.append(data_info)
                break
        return dataset_files

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index: int):
        img_data = self.images_data[index]
        img_path = str(Path(img_data.image, os.listdir(img_data.image)[0]))
        img = cv2.imread(img_path)[:, :, ::-1]

        if self.transform is not None:
            temp_img = self.transform(image=img)['image']
            temp_img /= 255.0

        return {
            'image': temp_img,
            'label': img_data.label
        }
