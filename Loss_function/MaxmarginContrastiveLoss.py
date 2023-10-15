import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxMarginContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(MaxMarginContrastiveLoss, self).__init__()
        self.margin_id = margin[0]
        self.margin_type = margin[1]

    def forward(self, y, z):
        labels_id = y[:, 0]
        labels_type = y[:, 1]

        # Compute pair-wise distance matrix
        D = pdist_euclidean(z)
        d_vec = D.view(-1, 1)

        # Make contrastive labels
        yid_contrasts = get_contrast_batch_labels(labels_id)
        ytype_contrasts = get_contrast_batch_labels(labels_type)

        loss = F.margin_ranking_loss(d_vec, yid_contrasts, margin=self.margin_id) + \
               F.margin_ranking_loss(d_vec, ytype_contrasts, margin=self.margin_type)

        return torch.mean(loss)

def pdist_euclidean(A):
    r = torch.sum(A*A, dim=1)
    r = r.view(-1, 1)
    D = r - 2*torch.matmul(A, torch.transpose(A, 0, 1)) + torch.transpose(r, 0, 1)
    return torch.sqrt(D)

def get_contrast_batch_labels(y):
    y_col_vec = y.view(-1, 1).float()
    D_y = pdist_euclidean(y_col_vec)
    d_y = D_y.view(-1, 1)
    y_contrasts = (d_y == 0).int()
    return y_contrasts