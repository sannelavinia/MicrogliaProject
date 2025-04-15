import torch
from utils import random_crop_image_and_label, get_transforms
import numpy as np
import cv2
import os

class MicrogliaDataset(torch.utils.data.Dataset):
    
    def __init__(self, images_dir, labels_dir, crop_size=(256, 256), data_augmentation=False, clahe=True, multiclass=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.clahe = clahe
        self.multiclass = multiclass
        
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

        if self.multiclass:
            # Read images
            image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)  # Read as RGB
            label = cv2.imread(label_path, cv2.IMREAD_COLOR_RGB)  # Read as RGB (in reality it's on one channel with values 0,1,2)

            # Single out correct channel, containing 0=background, 1=nonfoamy, 2=foamy
            label = label[:,:,2]

            # Expand label dimension to (H, W, 1)
            # label = np.expand_dims(label, axis=-1)
            label = label.astype(np.int64) 

        else:
            # Read images
            image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)  # Read as RGB
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

            # Normalize binary image from range (0-255) to (0.0-1.0)
            label = label.astype(np.float64) / np.float64(255.0)

            # Expand label dimension to (H, W, 1)
            label = np.expand_dims(label, axis=-1)

        # Apply histogram equalisation using CLAHE
            if self.clahe:
                clahe_application = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                l_clahe = clahe_application.apply(l)
                lab_clahe = cv2.merge((l_clahe, a, b))
                image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        # Convert to tensors
        if self.multiclass:
            image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW
            label_tensor = torch.tensor(label.transpose(2, 0, 1), dtype=torch.float32)  # (1, H, W)

        # Apply random crop
        image_out, label_out = random_crop_image_and_label(image_tensor, label_tensor, size=self.crop_size)

        # Apply data augmentation
        if self.data_augmentation:
            image_out, label_out = get_transforms(image_out, label_out)

        return image_out, label_out
