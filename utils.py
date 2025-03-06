import torch
from torchvision.transforms.v2 import functional as F

# Define custom Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.data_augmentation = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensors with correct dtype 
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW, Float
        label = torch.tensor(label.transpose(2, 0, 1), dtype=torch.float32)  # Add channel dimension (1, H, W)

        # Apply the custom random crop
        image_out, label_out = random_crop_image_and_label(image, label, size=(256, 256))
        
        return image_out, label_out

# Custom random crop function
def random_crop_image_and_label(image, label, size):
    from torchvision.transforms import RandomCrop
    # Get crop parameters
    i, j, h, w = RandomCrop.get_params(image, output_size=size)

    # Apply the crop to both image and label
    cropped_image = F.crop(image, i, j, h, w)
    cropped_label = F.crop(label, i, j, h, w)

    return cropped_image, cropped_label
