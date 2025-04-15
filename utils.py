from torchvision.transforms.v2 import functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as TF
import os

def visualize_predictions(model, val_loader, device, multiclass, num_images=10, show=False, save=False):
    model.eval()  # Set model to evaluation mode
    images_shown = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Get predictions

            # Move tensors to CPU and convert to NumPy
            images = images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert (C, H, W) → (H, W, C)

            if multiclass:
                labels = labels.cpu().numpy()

                predictions = torch.argmax(outputs, dim=1)
                predictions = predictions.cpu().numpy()

            else:
                labels = labels.cpu().numpy().squeeze(1)  # Remove channel dim

                # Convert predictions to binary (threshold = 0.5)
                predictions = (outputs > 0.5).float()
                predictions = predictions.cpu().numpy().squeeze(1)  # Remove channel dim

            # Plot images
            for i in range(min(num_images, len(images))):
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                
                ax[0].imshow(images[i].astype(np.uint8))  # Original Image
                ax[0].set_title("Original Image")
                
                ax[1].imshow(labels[i], cmap="viridis")  # Ground Truth
                ax[1].set_title("Ground Truth")
                
                ax[2].imshow(predictions[i], cmap="viridis")  # Model Prediction
                ax[2].set_title("Model Prediction")

                for a in ax:
                    a.axis("off")  # Hide axis ticks

                if show:
                    plt.show()

                if save:
                    os.makedirs(save, exist_ok=True)
                    image_name = f"image_{images_shown + 1}.png"
                    save_path = os.path.join(save, image_name)
                    plt.savefig(save_path)  # Save the image
                    plt.close(fig)  # Close the plot to prevent overlapping images

                images_shown += 1
                if images_shown >= num_images:
                    return  # Stop after showing the desired number of images

# Custom random crop function
def random_crop_image_and_label(image, label, size):
    from torchvision.transforms import RandomCrop
    # Get crop parameters
    i, j, h, w = RandomCrop.get_params(image, output_size=size)

    # Apply the crop to both image and label
    cropped_image = F.crop(image, i, j, h, w)
    cropped_label = F.crop(label, i, j, h, w)

    return cropped_image, cropped_label

def get_transforms(image, label):
    # Random transformations (using pytorch transorms didn't give the same transformations to label and image, so it had to be done manually)
    if torch.rand(1) > 0.5:  # Random Horizontal Flip
        image = TF.hflip(image)
        label = TF.hflip(label)

    if torch.rand(1) > 0.5:  # Random Vertical Flip
        image = TF.vflip(image)
        label = TF.vflip(label)

    return image, label 



