import torch
from images import load_images, view_results
from utils import Dataset
import matplotlib.pyplot as plt
import numpy as np
from dice_loss import DiceLoss

# Load U-NET model
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=False)

# Image data paths
images_dir = './images'
labels_dir = './labels'

# Load in images
    # Shape = (number of images, x size, y size, channels)
images_np, labels_np = load_images(images_dir, labels_dir, data_augmentation=True)

# Initialize dataset
dataset = Dataset(images_np, labels_np)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

## TRAINING THE MODEL ##
def train_model(model, train_loader, optimizer, loss_fn, epochs=20):
    model.train() # prepare the model to be trained

    for epoch in range(epochs):
        epoch_loss = 0.0

        for images, labels in train_loader:
            # Forward pass, make predictions for one batch
            outputs = model(images)

            # Compute loss between prediction and ground truth labels
            loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad() # set gradients to zero
            loss.backward() # computes gradients
            optimizer.step() # adjusts weights

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# loss_fn = torch.nn.BCELoss()
loss_fn = DiceLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

train_model(model, train_dataloader, optimizer, loss_fn, epochs=200)

# Save the trained model
model_save_path = './unet_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Visualize results (and save) for a sample image
for img in range(len(dataset)):
    view_results(model, dataset, idx=img, save_path=f"./results-augmentation-200epochs/restult-{img}.png", show=False)

# view_results(model, dataset, idx=0, show=True)
# view_results(model, dataset, idx=1, show=True)
# view_results(model, dataset, idx=2, show=True)
# view_results(model, dataset, idx=3, show=True)


            
