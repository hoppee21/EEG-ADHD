import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class EEGNet_encoder(pl.LightningModule):
    
    def __init__(self, Chans=56, Samples=385, kernLength=256, dropoutRate=0.25, F1=4, D=2, F2=8, norm_rate=0.25):
        super(Encoder, self).__init__()
