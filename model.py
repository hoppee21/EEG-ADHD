import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

class EEGNet_encoder(pl.LightningModule):
    
    def __init__(self, Chans=56, Samples=385, kernLength=256, dropoutRate=0.25, F1=4, D=2, F2=8, norm_rate=0.25):
        super(EEGNet_encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding=(0, kernLength//2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, kernel_size=(1, 32), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 2))
        self.dropout1 = nn.Dropout2d(p=dropoutRate)

        self.sep_conv = nn.Conv2d(F1 * D, 16, kernel_size=(1, 32), padding=(0, 32//2), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout2d(p=dropoutRate)

        self.depthwise_conv2 = nn.Conv2d(16, 16 * D, kernel_size=(28, 1), groups=16, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(16 * D)
        self.elu3 = nn.ELU()
        self.avgpool3 = nn.AvgPool2d((4, 8))

        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.sep_conv(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = self.depthwise_conv2(x)
        x = self.batchnorm4(x)
        x = self.elu3(x)
        x = self.avgpool3(x)

        x = self.flatten(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer