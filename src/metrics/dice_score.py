import torch

def calculate_dice(predictions, targets, threshold=0.5):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    dice = (2. * intersection + 1e-6) / (predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6)
    return dice.mean()