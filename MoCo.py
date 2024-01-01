import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


class MoCo(pl.LightningModule):
    def __init__(self, base_encoder, train_x, train_y, dim=128, K=9000, m=0.999, T=0.07, mlp=False):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.train_x = train_x
        self.train_y = train_y

        self.contrastive_loss = nn.CrossEntropyLoss()

        # create the encoders
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        print("Initializing queue")
        self.register_buffer("queue1", torch.randn(dim, K))
        self.register_buffer("queue2", torch.randn(dim, K))
        self.register_buffer("queue3", torch.randn(dim, K))

        self.queue1 = nn.functional.normalize(self.queue1, dim=0)
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.queue3 = nn.functional.normalize(self.queue3, dim=0)

        self.register_buffer("queue1_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue3_ptr", torch.zeros(1, dtype=torch.long))


    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def _dequeue_and_enqueue(self, keys, label):
        true_indices = torch.argmax(label, dim=1)
        if true_indices == 0:
            ptr = int(self.queue1_ptr)
            batch_size = keys.shape[0]

            assert self.K % batch_size == 0

            self.queue1[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
            self.queue1_ptr[0] = ptr
        elif true_indices == 1:
            ptr = int(self.queue2_ptr)
            batch_size = keys.shape[0]

            assert self.K % batch_size == 0

            self.queue2[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
            self.queue2_ptr[0] = ptr
        elif true_indices == 2:
            ptr = int(self.queue3_ptr)
            batch_size = keys.shape[0]

            assert self.K % batch_size == 0

            self.queue3[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
            self.queue3_ptr[0] = ptr
       

    def forward(self, im_q, im_k, label):
        q = self.encoder_q.forward(im_q)
        q = nn.functional.normalize(q, dim=1)

        true_indices = torch.argmax(label, dim=1)

        if true_indices == 0:
            queue = torch.cat([self.queue2, self.queue3], dim=1)
        elif true_indices == 1:
            queue = torch.cat([self.queue1, self.queue3], dim=1)
        elif true_indices == 2:
            queue = torch.cat([self.queue1, self.queue2], dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(2)

        self._dequeue_and_enqueue(k, label)

        return logits, labels

    # def training_step(self, batch, batch_idx):
    #     combined_data, label = batch

    #     split_size = 56
    #     im_q, im_k = torch.split(combined_data, split_size, dim=2)

    #     logits, labels = self.forward(im_q, im_k, label)
    #     loss = self.contrastive_loss(logits, labels)

    #     self.log('train_loss', loss, logger=True)

    #     return loss

    def training_step(self, batch, batch_idx):
        combined_data, labels = batch

        total_loss = 0
        for i in range(combined_data.size(0)):
            # Extract the data and label for the current item
            data_item = combined_data[i].unsqueeze(0)  # Add batch dimension
            label_item = labels[i].unsqueeze(0)  # Add batch dimension

            split_size = 56
            im_q, im_k = torch.split(data_item, split_size, dim=2)

            logits, label = self.forward(im_q, im_k, label_item)
            loss = self.contrastive_loss(logits, label)

            # Sum the loss
            total_loss += loss

        # Calculate the average loss
        avg_loss = total_loss / combined_data.size(0)

        self.log('train_loss', avg_loss, logger=True)

        return avg_loss
    
    def train_dataloader(self):
        dataset = TensorDataset(self.train_x, torch.LongTensor(self.train_y))
        train_loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)
        return train_loader

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.03, momentum=0.9)
        return optimizer




