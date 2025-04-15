import torch
from dataset import MicrogliaDataset
from torch.utils.data import random_split, DataLoader
from unet import UNet
from loss import DiceLoss
import optuna

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters suggested by Optuna
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    augmentation = trial.suggest_categorical("augmentation", [True, False])

    # Load dataset
    dataset = MicrogliaDataset(images_dir='./images',
                               labels_dir='./labels',
                               crop_size=(256, 256),
                               data_augmentation=augmentation)
   
    # Split dataset
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    model = UNet().to(device)

    # Loss function and optimizer
    #loss_fn = torch.nn.BCELoss()
    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop (short for tuning)
    num_epochs = 250  # Short training for tuning
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
       
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
       
        val_loss /= len(val_loader)

        # Optuna optimization
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
   
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100000)

    print(f"Best hyperparameters: {study.best_params} Best value: {study.best_value}")