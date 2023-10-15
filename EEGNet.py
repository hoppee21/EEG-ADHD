import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_metric_learning import losses
# from Loss_function import MaxMarginContrastiveLoss

class EEGNet_encoder(pl.LightningModule):
    
    def __init__(self, Chans=56, Samples=385, kernLength=256, dropoutRate=0.25, F1=4, D=2, F2=8, norm_rate=0.25, nb_classes=2):
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

        # Fully connected layer
        self.fc = nn.utils.weight_norm(nn.Linear(F1 * Samples * 8, nb_classes))

        # self.loss_function = MaxMarginContrastiveLoss(margin=(1.0, 3.0))
        self.loss_function = losses.SupConLoss(temperature=0.01)
    
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

        # Fully connected layer with weight normalization
        x = self.fc(x)

        # Softmax activation
        x = F.softmax(x, dim=1)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)

        # Log the loss for visualization in TensorBoard
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        # Split validation data into K folds
        k_folds = 10
        fold_size = len(val_batch) // k_folds
        fold_losses = []

        for fold in range(k_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size
            val_fold = Subset(val_batch, range(start_idx, end_idx))

            # Forward pass
            x, y = zip(*val_fold)
            x = torch.stack(x)
            y = torch.stack(y)
            y_hat = self(x)

            # Calculate loss for the fold
            fold_loss = self.loss_function(y, y_hat)
            fold_losses.append(fold_loss.item())

        # Calculate the average loss across folds
        avg_loss = sum(fold_losses) / k_folds

        # Log the average loss for visualization in TensorBoard
        self.log('val_loss', avg_loss, prog_bar=True, logger=True)

        return avg_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
