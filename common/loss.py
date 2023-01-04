#Dice损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.shape == target.shape, "the size of predict and target must be equal."
        N = predict.shape[0]

        pre = torch.sigmoid(predict).view(N, -1)
        tar = target.view(N, -1)

        intersection = (pre * tar).sum(-1)
        union = (pre + tar).sum(-1)

        dice = torch.mean(2 * (intersection + self.epsilon) / (union + self.epsilon))
        score = 1 - dice

        return score