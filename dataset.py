import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from augmentation import cutmix
import torch


def colormap_to_label(color_map):
    """build a mapping from RGB to label class index

    Args:
        color_map (list[list]): k*3, dataset category color list, K is the number of class

    Returns:
        tensor: a mapping from RGB to label class index
    """
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(color_map):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


def label_indices(mask, colormap2label):
    """map RGB values in labels to their class indices

    Args:
        mask (array): C*H*W, segmentation mask
        colormap2label (_type_): _description_

    Returns:
        tensor: H*W, single channel mask
    """
    mask = mask.permute(1, 2, 0).numpy().astype("long")
    idx = (mask[:, :, 0] * 256 + mask[:, :, 1]) * 256 + mask[:, :, 2]
    return colormap2label[idx]


def get_img(path):
    img = Image.open(path)
    return np.asarray(img.convert("RGB"))


def get_paths(path, phase):
    """get list of train data paths or test data paths

    Args:
        path (string): train data directory or test data directory
        phase (string): which data you want to use("train" or "test")

    Returns:
        phase == "train":
            tuple(list): (train_img_paths, train_mask_paths)
                train_img_paths: image paths
                train_mask_paths: mask paths
        phase == "test":
            list: image paths
    """
    if phase == "train":
        train_img_paths = [
            os.path.join(path, "img", d)
            for d in os.listdir(os.path.join(path, "img"))
            if os.path.isfile(os.path.join(path, "img", d))
        ]
        train_mask_paths = [
            os.path.join(path, "mask", d)
            for d in os.listdir(os.path.join(path, "mask"))
            if os.path.isfile(os.path.join(path, "mask", d))
        ]
        train_img_paths.sort(
            key=lambda x: int(x[len(os.path.join(path, "img")) + 1 : -4])
        )
        train_mask_paths.sort(
            key=lambda x: int(x[len(os.path.join(path, "mask")) + 1 : -4])
        )
        return train_img_paths, train_mask_paths
    elif phase == "test":
        test_img_paths = [
            os.path.join(path, "img", d)
            for d in os.listdir(os.path.join(path, "img"))
            if os.path.isfile(os.path.join(path, "img", d))
        ]
        test_img_paths.sort(
            key=lambda x: int(x[len(os.path.join(path, "img")) + 1 : -4])
        )
        return test_img_paths


class Dataset(Dataset):
    def __init__(self, path, color_map, transform=None, phase="train"):
        """make train pytorch dataset

        Args:
            train_img (list[array]): list of train image, list[H*W*C]
            train_mask (list[array]): list of train mask, list[H*W*C]
            color_map (list[list]): color category of mask
            transform ((array, array), optional): receive train image and single channel mask. Defaults to None.
            phase (string): which data you want to use("train" or "test")
        """
        assert phase in ["train", "test"]
        self.path = path
        self.colormap2label = colormap_to_label(color_map)
        self.transform = transform
        self.phase = phase
        if phase == "train":
            self.img_paths, self.mask_paths = get_paths(self.path, phase)
        elif phase == "test":
            self.img_paths = get_paths(self.path, phase)

    def __getitem__(self, index):
        """

        Args:
            index (int): Index

        Returns:
            phase == "train":
                tuple(tensor): (img, label)
                    img: C*H*W, image
                    label: H*W, single channel mask
            phase == "test":
                tuple(tensor): (img, shape)
                    img: C*H*W, image
                    shape: [2], before resize image's shape
        """
        # solve dataloader num_workers bug
        import cv2

        cv2.setNumThreads(0)
        if self.phase == "train":
            img, mask = get_img(self.img_paths[index]), get_img(self.mask_paths[index])
            if self.transform is not None:
                augments = self.transform(image=img, mask=mask)
                img, mask = augments["image"], augments["mask"]
            img, mask = cutmix(img, mask, 3, 10, 50, 0.95)
            label = label_indices(mask, self.colormap2label)
            return img, label
        elif self.phase == "test":
            img = get_img(self.img_paths[index])
            shape = np.array([img.shape[0], img.shape[1]])
            if self.transform is not None:
                augments = self.transform(image=img)
                img = augments["image"]
            return img, shape

    def __len__(self):
        return len(self.img_paths)
