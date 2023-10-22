import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from pytorch_metric_learning import losses, miners
# from Loss_function import MaxMarginContrastiveLoss

class EEGNet_encoder(pl.LightningModule):
    
    def __init__(self, train_x, train_y, valid_x, valid_y,alpha,learning_rate, kernLength=256, dropoutRate=0.25, F1=4, D=2, norm_rate=0.25, num_classes=3):
        super(EEGNet_encoder, self).__init__()

        # Optional: Apply gradient clipping
        self.norm_rate = norm_rate
        self.miner = miners.MultiSimilarityMiner()
        self.alpha = alpha
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, kernel_size=(1, 32), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 2))
        self.dropout1 = nn.Dropout2d(p=dropoutRate)

        self.sep_conv = nn.Conv2d(F1 * D, 16, kernel_size=(1, 32), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        # self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout2d(p=dropoutRate)

        self.depthwise_conv2 = nn.Conv2d(16, 16 * D, kernel_size=(28, 1), groups=16, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(16 * D)
        # self.elu3 = nn.ELU()
        self.avgpool3 = nn.AvgPool2d((4, 8))

        self.flatten = nn.Flatten()

        # Fully connected layer
        self.dense = nn.Linear(448, num_classes)

        self.loss = nn.BCELoss()
        # self.loss_function = MaxMarginContrastiveLoss(margin=(1.0, 3.0))
        # self.loss_function = losses.SupConLoss(temperature=0.01)
        self.loss_function = losses.TripletMarginLoss()
    
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
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = self.depthwise_conv2(x)
        x = self.batchnorm4(x)
        x = self.elu(x)
        x = self.avgpool3(x)

        x = self.flatten(x)

        # Fully connected layer
        x = self.dense(x)

        # Softmax activation
        # x = F.softmax(x, dim=1)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        hard_pairs = self.miner(y_hat, torch.argmax(y.float(),dim=1))
        loss_metric = - self.loss_function(y_hat, torch.argmax(y.float(), dim=1), hard_pairs) 

        y_pred = F.softmax(y_hat, dim=1) 
        loss_classify = self.loss(y_pred, y.float())

        alpha = self.alpha
        loss = alpha * loss_metric + (1 - alpha) * loss_classify
        # Log the loss for visualization in TensorBoard
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True,sync_dist=True, logger=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss(y_hat, y.float())


        # Log the average loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, logger=True)

        return loss
    
    def train_step_end(self, training_step_outputs):
        
        if self.norm_rate is not None:
            clip_grad_norm_(self.parameters(), self.norm_rate)

    def train_dataloader(self):
        dataset = TensorDataset(self.train_x, torch.LongTensor(self.train_y))
        train_loader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = TensorDataset(self.valid_x, torch.LongTensor(self.valid_y))
        val_loader = DataLoader(val_dataset, batch_size=16, num_workers=8, shuffle=False)
        return val_loader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=6, verbose=1,
                                                               mode='min', cooldown=0, min_lr=10e-7)
        optimizer_dict = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer_dict
