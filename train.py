import torch
from dataset import MicrogliaDataset
from torch.utils.data import random_split, DataLoader
from unet import UNet
from loss import DiceLoss
import argparse
import time
from utils import visualize_predictions

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load full dataset
    dataset = MicrogliaDataset(images_dir='./images', 
                               labels_dir='./labels', 
                               crop_size=(args.image_size, args.image_size),
                               data_augmentation=args.augmentation)
    
    # Split dataset
    train_size = int(0.85 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    torch.manual_seed(args.seed)  # Ensures reproducibility
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load model
    model = UNet()
    model.to(device)

    # Loss function and optimizer
    # loss_fn = DiceLoss()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    ### Training loop ###
    for epoch in range(args.epochs):
        model.train() # prepare model for training
        train_loss = 0.0

        start_time = time.time() # keep track of time

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass, make predictions for one batch
            outputs = model(images)

            # Compute loss between prediiction and ground truth labels
            loss = loss_fn(outputs, labels)

            # Backward pass, compute gradient
            optimizer.zero_grad() # set gradient to zero
            loss.backward() # compute gradient
            optimizer.step() # adjust weights

            train_loss += loss.item()

        train_loss /= len(train_loader) 

        ### Validation loop ###
        model.eval() 
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
        
        val_loss /= len(val_loader)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
    
    # After training, visualize some predictions
    visualize_predictions(model, val_loader, device, num_images=100, show=False, save="results_validation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4, help="Seed set before splitting dataset into validation & training")
    parser.add_argument("--augmentation", type=bool, default=True, help="Whether or not augmentation is applied to dataset")
    parser.add_argument("--epochs", type=int, default=800, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=256, help="Image crop size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    args = parser.parse_args()
    main(args)

    