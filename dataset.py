import torch
from utils import random_crop_image_and_label, get_transforms
import numpy as np
import cv2
import os

class MicrogliaDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, crop_size=(256, 256), data_augmentation=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get file paths
        image_file = self.image_files[idx]
        label_file = self.label_files[idx]
        
        img_path = os.path.join(self.images_dir, image_file)
        label_path = os.path.join(self.labels_dir, image_file.replace('.ome.tif', '-labels.png'))

        # Read images
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)  # Read as RGB
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        
        # Expand label dimension to (H, W, 1)
        label = np.expand_dims(label, axis=-1)

        # Normalize the label (0 -> 1, 1 -> 0) as QuPath reverses labels
        if np.max(label) == np.min(label):
            label_normalized = np.zeros_like(label)  # Set to all 0s (or 1s, depending on your needs)
        else:
            label_normalized = (label - np.min(label)) / (np.max(label) - np.min(label))

        label_binary = (label_normalized > 0.5).astype(np.uint8)
        label_reversed = 1 - label_binary

        # Convert to tensors
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW
        label_tensor = torch.tensor(label_reversed.transpose(2, 0, 1), dtype=torch.float32)  # (1, H, W)

        # Apply random crop
        image_out, label_out = random_crop_image_and_label(image_tensor, label_tensor, size=self.crop_size)

        # Apply data augmentation
        if self.data_augmentation:
            image_out, label_out = get_transforms(image_out, label_out)

        return image_out, label_out
