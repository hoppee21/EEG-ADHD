import torchvision.models as models
import torch
import torch.nn as nn

class Resnet50(nn.Module):
    '''
    Resnet 50 adapted for MoCo as an encoder.

    Args:
        output_dim (int): Dimension of the output feature vector.
        pretrained (bool): If True, use pretrained weights for the base ResNet50 model.
    '''
    def __init__(self, output_dim=128, pretrained=False):
        super(Resnet50, self).__init__()

        # Load the ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace the fully connected layer of ResNet50
        # The feature dimension should match the input of the contrastive loss or the queue
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        # Pass input through ResNet50
        features = self.resnet(x)
        return features