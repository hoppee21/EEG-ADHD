import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from Model import EEGNet_encoder

class Classifier(pl.LightningModule):
    def __init__(self, train_x, train_y, valid_x, valid_y, input_size=128, num_classes=3):
        super(Classifier, self).__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.encoder = EEGNet_encoder()

        self.load_encoder_weights()
        self.freeze_encoder()

        self.fc = nn.Linear(input_size, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = 0.001
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_encoder_weights(self):
        encoder_path = "/home/yinhuiwang/EEG/Moco_weights.pth"
        encoder_weights = torch.load(encoder_path, map_location='cpu')
        # Ensure the correct key is used
        encoder_q_state_dict = {key.replace('encoder_q.', ''): value for key, value in encoder_weights.items() if 'encoder_q' in key}

        # Load the original fc layer weights and the modified fc layer weights
        model_dict = self.encoder.state_dict()
        for key, value in encoder_q_state_dict.items():
            if 'fc' in key:
                fc_key = key.replace('fc.', 'fc')
                if fc_key in model_dict:
                    model_dict[fc_key] = value

    def forward(self, x):
        x = self.encoder(x)
        # Assuming x is the output from your encoder
        out = self.fc(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x) 
        loss = self.loss(y_hat, y.float())

        # Log the loss for visualization in TensorBoard
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True, logger=True)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        y_hat = F.softmax(y_hat, dim=1)
        
        # Calculate accuracy
        _, predicted = torch.max(y_hat, dim=1)
        true_indices = torch.argmax(y, dim=1)

        # Calculate accuracy
        correct = (predicted == true_indices).sum().item()
        total = y.size(0)
        accuracy = correct / total
        # Log the accuracy
        self.log('val_accuracy', accuracy, sync_dist=True, logger=True, on_epoch=True, prog_bar=True)

        return accuracy
    
    def train_dataloader(self):
        dataset = TensorDataset(self.train_x, torch.LongTensor(self.train_y))
        train_loader = DataLoader(dataset, batch_size=8, num_workers=8, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = TensorDataset(self.valid_x, torch.LongTensor(self.valid_y))
        val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4, shuffle=False)
        return val_loader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=3, verbose=True,
                                                               mode='min', cooldown=0, min_lr=10e-7)
        optimizer_dict = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
        return optimizer_dict