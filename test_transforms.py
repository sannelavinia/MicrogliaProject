import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import RandomCrop
import matplotlib.pyplot as plt

# Custom transformation for image and label
def random_crop_image_and_label(image, label, size):
    # Get crop parameters
    i, j, h, w = RandomCrop.get_params(image, output_size=size)
    
    # Apply the crop to both image and label
    cropped_image = F.crop(image, i, j, h, w)
    cropped_label = F.crop(label, i, j, h, w)
    return cropped_image, cropped_label

# Load in the image and label
image = cv2.imread('./train/0cdf5b5d0ce1_01.jpg', cv2.IMREAD_COLOR)
with Image.open('./trainannot/0cdf5b5d0ce1_01_mask.gif') as gif:
    label = np.array(gif.convert('L'))  # Convert to grayscale

# Preprocess
image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW
label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1, H, W)

print(image.shape)
print(label.shape)

# Apply the transformation
out_img, out_label = random_crop_image_and_label(image, label, size=(256, 256))

print(out_img.shape)
print(out_label.shape)

# Convert tensors back to numpy arrays for visualization
image_np = out_img.numpy().transpose(1, 2, 0).astype(np.uint8)  # CHW -> HWC
label_np = out_label.squeeze().numpy()  # Remove channel dimension

print(image_np.shape)
print(label_np.shape)
# Visualize
plt.imshow(image_np)
plt.show()

plt.imshow(label_np, cmap='gray')
plt.show()
