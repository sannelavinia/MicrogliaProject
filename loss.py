import torch

# Defining a custom dice loss function in PyTorch
class DiceLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-7):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon # very small constant to solve instances of zero division, default is 1e-7

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size() # make sure labels have same size

        y_pred = torch.flatten(y_pred) # Flatten tensors to make 1D arrays, simplifies calculations
        y_true = torch.flatten(y_true)

        intersection = torch.sum(y_pred * y_true) # calculate intersaction of true and predicted labels
        total = torch.sum(y_pred) + torch.sum(y_true) # calculate total amount of voxels that make up both the true and predicted labels
        dice_score = (2 * intersection + self.epsilon) / (total + self.epsilon)

        return 1 - dice_score