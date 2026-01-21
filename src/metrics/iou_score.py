import torch

def calculate_iou(predictions, targets, threshold=0.5):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()