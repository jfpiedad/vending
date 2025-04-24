import os
from glob import glob

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms


class OpenImagesV7Dataset(torch.utils.data.Dataset):
    def __init__(self, labels_path: str, image_size: int, augment: A.Compose = None):
        """Initialize the Dataset for BlazeFace. We are going to use a subset of the Open Images V7 by Google.

        Args:
            labels_path (str): Path to the labels directory.
            image_size (int): Size to which images will be resized.
            augment (A.Compose, optional): Albumentations augmentation pipeline. Defaults to None.
        """
        self.labels = list(sorted(glob(f"{labels_path}/*")))
        self.labels = [x for x in self.labels if os.stat(x).st_size != 0]
        self.augment = augment
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((image_size, image_size)),
            ]
        )
        self.image_size = image_size

    def __getitem__(self, idx: int) -> tuple:
        """Get item method, including image and labels loading and preprocessing.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Transformed image and target bounding boxes.
        """
        # load image
        img_path = self.labels[idx].replace("labels", "images")[:-3] + "jpg"
        img = self.load_image(img_path)
        rescale_output = self.resize_and_pad(img, self.image_size)
        img = rescale_output["image"]
        # Read and convert labels
        target = self.read_and_convert_labels(self.labels[idx], rescale_output)
        # Apply data augmentation
        if self.augment is not None:
            augmented = self.augment(image=img, bboxes=target)
            img = augmented["image"]
            target = np.array(augmented["bboxes"])

        return self.transform(img.copy()), np.clip(target, 0, 1)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.labels)

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load an image from a filename.

        Args:
            image_path (str): Path to the image.

        Returns:
            np.ndarray: Array of the image.
        """
        img = plt.imread(image_path)
        if len(img.shape) == 2 or img.shape[2] == 1:
            # Handle grayscale images
            img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] == 4:
            # Handle alpha
            img = img[:, :, :3]
        return img

    @staticmethod
    def read_and_convert_labels(labels_idx: str, rescale_output: dict) -> np.ndarray:
        """Read and convert labels from YOLO format to x1, y1, x2, y2 format.

        Args:
            labels_idx (str): Path to the label file.
            rescale_output (dict): Rescaling output containing ratios and offsets.

        Returns:
            np.ndarray: Converted target bounding boxes.
        """
        annotations = pd.read_csv(labels_idx, header=None, sep=" ")
        labels = annotations.values[:, 0]
        yolo_bboxes = annotations.values[:, 1:]
        cx = yolo_bboxes[:, 0]
        cy = yolo_bboxes[:, 1]
        w = yolo_bboxes[:, 2]
        h = yolo_bboxes[:, 3]
        x1 = (cx - w / 2) * rescale_output["x_ratio"] + rescale_output["x_offset"]
        x2 = (cx + w / 2) * rescale_output["x_ratio"] + rescale_output["x_offset"]
        y1 = (cy - h / 2) * rescale_output["y_ratio"] + rescale_output["y_offset"]
        y2 = (cy + h / 2) * rescale_output["y_ratio"] + rescale_output["y_offset"]
        x1 = np.expand_dims(x1, 1)
        x2 = np.expand_dims(x2, 1)
        y1 = np.expand_dims(y1, 1)
        y2 = np.expand_dims(y2, 1)
        target = np.concatenate([x1, y1, x2, y2, labels.reshape(-1, 1)], axis=1).clip(
            0.0, 1.0
        )
        return target

    @staticmethod
    def resize_and_pad(img: np.ndarray, target_size: int = 128) -> dict:
        """Resize image to square target_size, and pad if needed to avoid deformation.

        Args:
            img (np.ndarray): Input image.
            target_size (int, optional): Target size for resizing. Defaults to 128.

        Returns:
            dict: Rescaled image and rescaling parameters.
        """
        if img.shape[0] > img.shape[1]:
            new_y = target_size
            new_x = int(target_size * img.shape[1] / img.shape[0])
        else:
            new_y = int(target_size * img.shape[0] / img.shape[1])
            new_x = target_size
        output_img = cv2.resize(img, (new_x, new_y))
        top = max(0, new_x - new_y) // 2
        bottom = target_size - new_y - top
        left = max(0, new_y - new_x) // 2
        right = target_size - new_x - left
        output_img = cv2.copyMakeBorder(
            output_img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(128, 128, 128),
        )
        # Compute labels values updates
        x_ratio = new_x / target_size
        y_ratio = new_y / target_size
        x_offset = left / target_size
        y_offset = top / target_size

        return {
            "image": output_img,
            "x_ratio": x_ratio,
            "x_offset": x_offset,
            "y_ratio": y_ratio,
            "y_offset": y_offset,
        }
