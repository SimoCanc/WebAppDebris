import torch
import torch.nn.functional as F

# Loss-function che penalizza la frammentazione e la non linearità delle maschere 
def regularization_loss(y_pred):
    dx = torch.abs(y_pred[:, :, 1:] - y_pred[:, :, :-1])
    dy = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    reg_loss = torch.mean(dx) + torch.mean(dy)
    return reg_loss

# Loss-function combinata, basata su Crossentropy-loss
def combined_loss(y_pred, y_true):
    ce_loss = F.cross_entropy(y_pred, y_true, weight=wgts)
    reg_loss = regularization_loss(y_pred)
    total_loss = ce_loss + reg_loss 
    return total_loss
