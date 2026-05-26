from torch.optim import AdamW,Adam
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights,resnet152,ResNet152_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append("../../")
from ctf_datasets.celeba.dataset import Celeba

class CelebaClassifier(pl.LightningModule):
    def __init__(self, attr, in_shape = (3, 64, 64), num_outputs = 1, lr=1e-3, context_dim=0,pretrain=True):
        super().__init__()
        self.variable = attr

        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()


        self.lr = lr
        self.variables = {"Smiling":0, "Eyeglasses":1}
        self.accociations = {"Smiling":None, "Eyeglasses":None}
        self.conditions = self.accociations[attr]
        self.attr = self.variables[attr] #select attribute
        in_channels = in_shape[0]

        self.num_outputs = num_outputs

        '''cnn layer implementation taken from https://openreview.net/forum?id=lZOUQQvwI3q'''
        if pretrain == True:
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            net = resnet50()
        num_features = net.fc.in_features
        modules = list(net.children())[:-1]
        self.cnn = torch.nn.Sequential(*modules)
        # new cls
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=num_features, out_features=1024),
        #     nn.ReLU(),
        #     #nn.Dropout(0.2),
        #     nn.Linear(in_features=1024, out_features=1),
        # )
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=1024, out_features=1),
        )
        # self.cnn = nn.Sequential(
        #                 nn.Conv2d(3, 16, 3, 1, 1),
        #                 nn.BatchNorm2d(16),
        #                 nn.ReLU(),
        #                 nn.Conv2d(16, 32, 3, 2, 1),
        #                 nn.BatchNorm2d(32),
        #                 nn.ReLU(),
        #                 nn.Conv2d(32, 32, 3, 1, 1),
        #                 nn.BatchNorm2d(32),
        #                 nn.ReLU(),
        #                 nn.Conv2d(32, 64, 3, 2, 1),
        #                 nn.BatchNorm2d(64),
        #                 nn.ReLU(),
        #                 nn.Conv2d(64, 64, 3, 1, 1),
        #                 nn.BatchNorm2d(64),
        #                 nn.ReLU(),
        #                 nn.Conv2d(64, 128, 3, 2, 1),
        #                 nn.BatchNorm2d(128),
        #                 nn.ReLU(),
        #                 nn.AdaptiveAvgPool2d(1),
        #             )

        # self.fc = nn.Sequential(
        #             nn.Linear(128, 128),
        #             nn.BatchNorm1d(128),
        #             nn.ReLU(),
        #             nn.Linear(128, self.num_outputs)
        #         )

    def forward(self, x, y=None):
        x = self.cnn(x)
        x = x.mean(dim=(-2, -1))  # avg pooling

        return self.fc(x)


    def training_step(self, batch):
        x, attrs_ = batch

        y = attrs_[:, self.attr] #select attribute to train

        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.type(torch.float32).view(-1, 1)) #applies sigmoid

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch):
        x, attrs_ = batch

        y = attrs_[:, self.attr]

        y_hat = self(x)

        loss = nn.BCEWithLogitsLoss()(y_hat, y.type(torch.float32).view(-1, 1)) #applies sigmoid
        val_f1 = self.f1_score(y_hat, y.type(torch.long).view(-1,1))
        val_acc = self.accuracy(y_hat, y.type(torch.long).view(-1,1))
        metrics = {'val_acc': val_acc,'val_f1': val_f1,'val_loss': loss,'val_metric':val_f1}
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=[0.9, 0.999])
        return optimizer

class Celeba_anticausal_Classifier(pl.LightningModule):
    def __init__(self, attr, in_shape=(3, 64, 64), num_outputs=1, lr=1e-3, pretrain=True):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrain"])  # keeps lr etc. in the ckpt
        self.variable = attr

        # metrics expect logits (we use BCEWithLogitsLoss)
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()

        self.lr = lr
        self.variables = {"Smiling": 0, "Eyeglasses": 1}
        self.accociations = {"Smiling":None, "Eyeglasses":None}
        self.conditions = self.accociations[attr]
        self.attr = self.variables[attr]
        in_channels = in_shape[0]
        self.num_outputs = num_outputs

        # backbone
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if pretrain else resnet50()
        num_features = net.fc.in_features
        modules = list(net.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # head
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=(-2, -1))         # global avg pool
        return self.fc(x)                # logits

    def training_step(self, batch, batch_idx):
        x, attrs = batch
        y = attrs[:, self.attr].to(torch.float32).view(-1, 1)
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, attrs = batch
        y = attrs[:, self.attr].to(torch.long).view(-1, 1)
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, y.to(torch.float32))
        val_f1 = self.f1_score(logits, y)
        val_acc = self.accuracy(logits, y)

        self.log_dict(
            {"val_loss": loss, "val_f1": val_f1, "val_acc": val_acc, "val_metric": val_f1},
            prog_bar=True, logger=True, on_epoch=True, batch_size=x.size(0)
        )
        return {"val_loss": loss, "val_f1": val_f1}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        # halve LR after 50 epochs without val_f1 improvement
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=50, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }