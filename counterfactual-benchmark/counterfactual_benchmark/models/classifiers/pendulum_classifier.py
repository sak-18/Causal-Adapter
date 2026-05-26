from torch.optim import Adam
#from germancredit.data.meta_data import attrs
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
import torch.nn.functional as F
import sys
sys.path.append("../../")
from ctf_datasets.morphomnist.dataset import MorphoMNISTLike

class PendClassifier(pl.LightningModule):
    def __init__(self, attr, in_shape = (3, 96, 96), width=8, num_outputs = 1, context_dim = 0 , lr=1e-3):
        super().__init__()
        self.variable = attr

        self.lr = lr
        self.variables = {"pendulum":0,"light":1,"shadow_length":2,"shadow_position":3}
        self.attr = self.variables[attr] #select attribute
        in_channels = in_shape[0]
        res = in_shape[1]
        s = 2 if res > 64 else 1

        '''CNN taken from https://github.com/Akomand/CausalDiffAE/blob/main/improved_diffusion/nn.py#L115'''
        hidden_dims = [16, 32, 32, 64, 64, 128]
        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc = nn.Linear(hidden_dims[-1]*4, 1)

    def forward(self, x, y=None):

        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        
        out = self.fc(result)
        return out


    def training_step(self, batch, batch_idx):
        x, attrs_ = batch

        y = attrs_[:, self.attr] #select attribute to train
        y_hat = self(x)

        loss = nn.MSELoss()(y_hat, y.type(torch.float32).view(-1, 1))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        #loss = nn.CrossEntropyLoss()(y_hat, y.type(torch.long))
        return loss


    def validation_step(self, batch, batch_idx):
        x, attrs_ = batch

        y = attrs_[:, self.attr] #select attribute to train
        y_hat = self(x)

        loss = nn.L1Loss()(y_hat, y.type(torch.float32).view(-1, 1))


        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_metric', loss, on_step=False , on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
