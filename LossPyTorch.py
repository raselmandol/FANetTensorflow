import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    # def forward(self, inputs, targets):
    #     inputs = torch.sigmoid(inputs)
    #     intersection = torch.sum(inputs * targets)
    #     dice = (2. * intersection + self.smooth) / (torch.sum(inputs) + torch.sum(targets) + self.smooth)
    #     return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = torch.sum(inputs * targets)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(inputs) + torch.sum(targets) + self.smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (0.5 * BCE) + (0.5 * dice_loss)
        return Dice_BCE
