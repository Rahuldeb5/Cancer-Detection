import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.shape[0], -1)
        target = target.reshape(target.shape[0], -1).float()

        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
