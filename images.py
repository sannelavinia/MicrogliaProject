import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import v2

# This file contains a function to load the images that will be used
# It also contains a function to view the results for image x in a given dataset

def load_images(images_dir, labels_dir, data_augmentation=False, data_augmentation_range=1):
    images, labels = [], []

    # Get sorted lists of image and label filenames
    image_files = sorted(os.listdir(images_dir))
    label_files = sorted(os.listdir(labels_dir))
    
    for image_file, label_file in zip(image_files, label_files):
        
        img_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, image_file.replace('.ome.tif', '-labels.png'))

        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB) # Read as RGB
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        # Add channel dimension to labels (this ensures shape is (height, width, 1))
        label = np.expand_dims(label, axis=-1)

        ### Convert 0 -> 1 and 1 -> 0, as after exporting from QuPath this isn't done correctly
        # Compute the minimum and maximum values dynamically
        min_val = np.min(label)
        max_val = np.max(label)

        # Normalize the array to the range [0, 1]
        label_normalized = (label - min_val) / (max_val - min_val)

        # Binarize the normalized array using a threshold (e.g., 0.5)
        label_binary = (label_normalized > 0.5).astype(np.uint8)

        # Assuming label_binary is your binary array with values 0 and 1
        # This needs to be done for microglia as QuPath reverses the labels for some reason
        label_reversed = 1 - label_binary

        images.append(image)
        labels.append(label_reversed)

        if data_augmentation:
            for augmentations in range(data_augmentation_range):
                # Set images to tensors to prepare for transformation
                image_tensor = torch.from_numpy(image).permute(2,0,1) # Convert to (C, H, W)
                label_tensor = torch.from_numpy(label_reversed).permute(2,0,1) # Convert to (1, H, W)

                # # Random transformations (using pytorch transorms didn't give the same transformations to label and image, so it had to be done manually)
                if torch.rand(1) > 0.5:  # Random Horizontal Flip
                    image_tensor = TF.hflip(image_tensor)
                    label_tensor = TF.hflip(label_tensor)

                if torch.rand(1) > 0.5:  # Random Vertical Flip
                    image_tensor = TF.vflip(image_tensor)
                    label_tensor = TF.vflip(label_tensor)

                image_np_augmented = image_tensor.permute(1,2,0).numpy().astype(np.uint8)
                label_np_augmented = label_tensor.permute(1,2,0).numpy().astype(np.uint8)

                images.append(image_np_augmented)
                labels.append(label_np_augmented)
    
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.uint8)

def view_results(model, dataset, idx=0, save_path=None, show = False):
    model.eval()
    image, label = dataset[idx]
    image = image.unsqueeze(0) # model expects a "batch" even when predicting for 1 image

    with torch.no_grad():
        output = model(image)
        predicted = (output > 0.5).float().squeeze().numpy()

    image_np = image.squeeze().numpy().transpose(1, 2, 0)
    label_np = label.squeeze().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(label_np, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted, cmap="gray")
    plt.title("Model Prediction")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()