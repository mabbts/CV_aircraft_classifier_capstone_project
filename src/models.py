import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.subplots as sub
import dash
from dash import dcc, html, Output, Input, State
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.trial import TrialState
#from optuna.integration import PyTorchIgnitePruningHandler

from functools import partial
import random
import os
import itertools
from PIL import Image
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torchvision
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision.models import vgg16_bn, resnet50, resnet18, efficientnet_b0, densenet121, ResNet50_Weights, ResNet18_Weights, VGG16_BN_Weights, DenseNet121_Weights, EfficientNet_B0_Weights
from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks, draw_keypoints
from torchvision import datasets
from torchvision.transforms import ToTensor, v2, ToPILImage
from torchvision.io import decode_image

# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
# from ignite.handlers import ModelCheckpoint, EarlyStopping

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from transformers import ViTForImageClassification, ViTImageProcessor, AutoModelForImageClassification, AutoImageProcessor, Trainer, TrainingArguments
from huggingface_hub import snapshot_download, hf_hub_download
import socket
import json
import sys
import io
import base64

# Simple CNN Backbone
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=30, attention_module=None):#, p_conv = 0.2, p_fc = 0.5
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 224x224 -> 224x224
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112
            #nn.Dropout2d(p_conv), # spatial dropout

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 112x112
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 56x56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
            #nn.Dropout2d(p_conv),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 28x28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # 14x14
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # B, 1024, 1, 1
        )
        self.attention = attention_module(1024) if attention_module else nn.Identity() 
        #self.dropout_fc = nn.Dropout(p_fc)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x) #Bx1024x1x1
        #if self.attention:
        x = self.attention(x) # or Identity
        x = x.view(x.size(0), -1) #Bx1024
        #x = self.dropout_fc(x)
        x = self.classifier(x)
        return x
    
#BCNN Module
class BCNN(nn.Module):
    def __init__(self, num_classes=30, attention_module=None):
        super(BCNN, self).__init__()

        # Custom feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # [B, 64, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 64, 112, 112]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B, 128, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 128, 56, 56]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# [B, 256, 56, 56]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                 # [B, 256, 1, 1]
        )
        
        self.attention = attention_module(256) if attention_module else nn.Identity()
        self.output_dim = 256
        self.fc = nn.Linear(self.output_dim * self.output_dim, num_classes)

    def forward(self, x):
        x = self.features(x)  # [B, 256, 1, 1]
        x = self.attention(x) # Apply attention if provided
        x = x.view(x.size(0), self.output_dim)  # [B, 256]

        # Bilinear pooling
        x = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))  # [B, 256, 256]
        x = x.view(x.size(0), -1)  # [B, 256*256]

        # Signed square root normalization
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)

        # L2 normalization
        x = F.normalize(x)

        # Classification
        x = self.fc(x)
        return x


# CAP Module
class CAP(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CAP, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        context = self.global_avg_pool(x)
        attention = self.fc1(context)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        return x * attention

# SE Module
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# CBAM Module
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_att))
        x = x * spatial_att

        return x


class CAPResNet(nn.Module):
    def __init__(self, num_classes=30, drop=0.0):
        super(CAPResNet, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.cap = CAP(in_channels=2048)
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.cap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ResNet50 with SE Block
class SEEffNet(nn.Module):
    def __init__(self, num_classes=30, drop=0.0):
        super(SEEffNet, self).__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity() # fc if resnet
        self.se = SEBlock(in_channels=1280) #2048 for resnet, 1280 for efficientnet, densenet121 = 1024, final layer also classifier
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.se(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# ResNet50 with CBAM Block
class CBAMResNet(nn.Module):
    def __init__(self, num_classes=30):
        super(CBAMResNet, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.cbam = CBAM(in_channels=2048)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.cbam(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ResNet50 with BCNN Block (Bilinear CNN)
class BCNNResNet(nn.Module):
    def __init__(self, num_classes=30):
        super(BCNNResNet, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() # Remove final fully connected layer
        self.output_dim = 2048 # resnet50 final feature depth
        self.classifier = nn.Linear(self.output_dim * self.output_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x) # extract features Bx2048
        x = x.unsqueeze(-1).unsqueeze(-1) # Bx2048x1x1
        x = x.view(x.size(0), self.output_dim) # Bx2048
        x = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)) # Bilinear pooling
        x = x.view(x.size(0), -1) # Bx2048*2048
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10) # signed square root normalization
        x = F.normalize(x) # L2 normalization
        x = self.classifier(x) # classification
        return x


# Loss functions
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
