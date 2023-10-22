import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

class SupervisedContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.01):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        
        # Compute logits
        logits = torch.matmul(feature_vectors_normalized, feature_vectors_normalized.t()) / self.temperature

        # Multi labels
        labels_id = labels[:, 0].view(-1, 1)
        labels_type = labels[:, 1].view(-1, 1)

        pair_loss = losses.NTXentLoss(logits, labels_id.squeeze(), temperature=self.temperature) + 2*losses.NTXentLoss(logits, labels_type.squeeze(), temperature=self.temperature)

        return pair_loss