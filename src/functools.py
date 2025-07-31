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

class FGVCAircraftDataset(Dataset):
    """
    A customised Fine Grained Visual Categorization Aircraft Dataset for manufacturer / family / variant annotation labels 
    similar to torchvision.datasets.FGVCAircraft dataset.
    
    Args:
        root_dir (str): path to  'fgvc-aircraft-2013b/data' folder
        split   (str): one of 'train', 'trainval', 'val' or 'test' 
        level   (str): one of 'manufacturer', 'family', or 'variant'.
        transform (callable, optional): a torchvision.transforms pipeline or Albumentation pipeline to apply to the images.
        return_label_str (bool): if True, __getitem__ returns (img, label_str) instead of (img, label_idx).
        use_cropped (bool): if True, __getitem__ returns cropped images using bounding box coordinates.
    """
    ALLOWED = {"manufacturer", "family", "variant"}
    
    def __init__(self, root= ROOT, split = "train", level = "manufacturer",
                 transform = None, return_class = False, cropped = False, album = False):
        assert level in self.ALLOWED, f"level must be one of {self.ALLOWED}"
        self.root = root
        self.split    = split
        self.level    = level
        self.transform = transform
        self.return_class = return_class
        self.cropped = cropped
        self.album = album

        # 1) read lines from images_{level}_{split}.txt
        class_file = os.path.join(root,
                                  f"images_{level}_{split}.txt")
        if not os.path.isfile(class_file):
            raise FileNotFoundError(f"{class_file} not found")

        samples = []
        with open(class_file, "r") as f:
            for ln in f:
                parts = ln.strip().split()
                img_id, *label_parts = parts
                class_str = " ".join(label_parts)
                samples.append((img_id, class_str))
        self.samples = samples

        # 2) build label→idx map
        self.classes = sorted({lbl for _, lbl in samples})
        self.class_to_idx = {lbl: i for i, lbl in enumerate(self.classes)}
        self.idx_to_class = {i: lbl for lbl, i in self.class_to_idx.items()}

        # 3) read bounding box coordinates from bounding_boxes_{split}.txt
        bbox_file = os.path.join(root, "images_box.txt")
        if not os.path.isfile(bbox_file):
            raise FileNotFoundError(f"{bbox_file} not found")

        self.bboxes = {}
        with open(bbox_file, "r") as f:
            for ln in f:
                parts = ln.strip().split()
                img_id = parts[0]
                bbox = tuple(map(int, parts[1:]))
                self.bboxes[img_id] = bbox

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, class_str = self.samples[idx]
        img_path = os.path.join(self.root, "images", f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        if self.cropped:
            xmin, ymin, xmax, ymax = self.bboxes[img_id]
            img = img.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            if not self.album:
                img = self.transform(img) #pytorch transoform
            else:
                aug = self.transform(image = img_np)
                img = aug['image'] # albumentations

        if self.return_class:
            return img, class_str

        class_idx = self.class_to_idx[class_str]
        return img, class_idx
    
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

def get_loaders(img_size = 224, batch_size = 32, annot = 'manufacturer'):
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]

    # train_tf = A.Compose([
    #     A.RandomResizedCrop((img_size, img_size)),  # A.Resize(img_size, img_size)
    #     A.HorizontalFlip(p=0.5),
    #     A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ])
    train_tf = A.Compose([
    A.RandomResizedCrop((img_size, img_size)),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
    
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.MotionBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(
            std_range=(0.02, 0.1),  # Adjusted for subtle noise
            mean_range=(0.0, 0.0),
            per_channel=True,
            noise_scale_factor=1.0,
            p=0.3
        )
    ], p=0.3),
    
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(8, 16),
        hole_width_range=(8, 16),
        fill=0,
        p=0.5
    ),
    
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
    ])
    test_tf = A.Compose([
        #A.LongestMaxSize(max_size=256)
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    
    train_dataset = FGVCAircraftDataset(root=ROOT, split='train', level = annot, transform=train_tf, 
                                        return_class=False, cropped=True, album = True)
    val_dataset   = FGVCAircraftDataset(root=ROOT, split='val', level = annot, transform=test_tf, 
                                        return_class=False, cropped=False, album = True)
    test_dataset   = FGVCAircraftDataset(root=ROOT, split='test', level = annot, transform=test_tf, 
                                         return_class=False, cropped=False, album = True)
    class_names = train_dataset.classes
    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, num_classes, class_names

def get_datasets(img_size = 224, batch_size = 32, annot = 'manufacturer'):
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]

    # train_tf = A.Compose([
    #     A.RandomResizedCrop((img_size, img_size)),  # A.Resize(img_size, img_size)
    #     A.HorizontalFlip(p=0.5),
    #     A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ])
    
    train_tf = A.Compose([
    A.RandomResizedCrop((img_size, img_size)),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
    
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.MotionBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(
            std_range=(0.02, 0.1),  # Adjusted for subtle noise
            mean_range=(0.0, 0.0),
            per_channel=True,
            noise_scale_factor=1.0,
            p=0.3
        )
    ], p=0.3),
    
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(8, 16),
        hole_width_range=(8, 16),
        fill=0,
        p=0.5
    ),
    
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
    ])

    test_tf = A.Compose([
        #A.LongestMaxSize(max_size=256)
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    train_dataset = FGVCAircraftDataset(root=ROOT, split='train', level = annot, transform=train_tf, 
                                        return_class=False, cropped=True, album = True)
    val_dataset   = FGVCAircraftDataset(root=ROOT, split='val', level = annot, transform=test_tf, 
                                        return_class=False, cropped=False, album = True)
    test_dataset   = FGVCAircraftDataset(root=ROOT, split='test', level = annot, transform=test_tf, 
                                         return_class=False, cropped=False, album = True)
    class_names = train_dataset.classes
    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_dataset, val_dataset, test_dataset

# For evaluation of valid set
def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k_preds = output.topk(k, dim=1)[1]
        correct = max_k_preds.eq(target.view(-1, 1).expand_as(max_k_preds))
        return correct.any(dim=1).float().mean().item()

# For evaluation of final test set

def compute_metrics(y_true, y_pred_logits, k=5):
    """
    Computes Top-1 and Top-k accuracy, and macro/micro F1 scores.
    
    Parameters:
    - y_true: Ground truth labels (list or numpy array)
    - y_pred_logits: Model output logits (tensor or numpy array)
    - k: Value for Top-k accuracy
    
    Returns:
    - Dictionary of metrics
    """
    # Convert logits to predicted labels
    y_pred_top1 = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred_logits.cpu().numpy() if isinstance(y_pred_logits, torch.Tensor) else y_pred_logits

    # Compute metrics
    top1_acc = top_k_accuracy_score(y_true_np, y_pred_np, k=1)
    topk_acc = top_k_accuracy_score(y_true_np, y_pred_np, k=k)
    f1_macro = f1_score(y_true_np, y_pred_top1, average='macro')
    f1_micro = f1_score(y_true_np, y_pred_top1, average='micro')

    return {
        'Top-1 Accuracy': top1_acc,
        f'Top-{k} Accuracy': topk_acc,
        'F1 Macro': f1_macro,
        'F1 Micro': f1_micro
    }


# Training and evaluation functions
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loss_lst = []
    acc_lst = []

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        #inputs, labels = cutmix_or_mixup(inputs, labels)#cutmix/mixup
        inputs, labels = inputs.to(device), labels.to(device)
        #hard_labels = labels.argmax(dim=1) #convert soft label to hard labels for cutmix or mixup

        optimizer.zero_grad()
        outputs = model(inputs)#
        loss = criterion(outputs, labels)#
        loss.backward()#
        optimizer.step()#
        # with autocast(device_type = 'cuda'):
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        

        running_loss += loss.item() * inputs.size(0)
        #hard_labels = labels.argmax(dim=1) #convert soft label to hard labels for cutmix or mixup
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        loss_lst.append(loss.item() * inputs.size(0))#
        acc_lst.append(predicted.eq(labels).sum().item() / labels.size(0))

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    #loss_lst.append(epoch_loss)
    #acc_lst.append(epoch_acc)
    return epoch_loss, epoch_acc, loss_lst, acc_lst

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    loss_lst = []
    acc_lst = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc = "Evaluation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            

            running_loss += loss.item() * inputs.size(0)
            #hard_labels = labels.argmax(dim=1) #convert soft label to hard labels for cutmix or mixup
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss_lst.append(loss.item() * inputs.size(0))#
            acc_lst.append(predicted.eq(labels).sum().item() / labels.size(0))

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    #loss_lst.append(epoch_loss)
    #acc_lst.append(epoch_acc)
    return epoch_loss, epoch_acc, all_preds, all_labels, loss_lst, acc_lst



# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def unnormalize(img_tensor, mean, std):
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return img_tensor

def visualize_predictions(model, test_dataset, num_samples=10, normalized=True):
    model.eval()
    samples = random.sample(range(len(test_dataset)), num_samples)
    # Dynamically calculate rows and columns
    cols = 5
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    axes = axes.flatten()

    inv_label_map = test_dataset.idx_to_class
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    with torch.no_grad():
        for idx, sample_idx in enumerate(samples):
            image, label = test_dataset[sample_idx]
            input_img = image.unsqueeze(0).to(device)
            output = model(input_img)
            _, pred = torch.max(output, 1)

            # Unnormalize if needed
            if normalized:
                image = unnormalize(image.clone(), mean, std)

            img_disp = image.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[idx].imshow(img_disp)
            axes[idx].set_title(f"Pred: {inv_label_map[pred.item()]}\nActual: {inv_label_map[label]}")
            axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

# Model selector
def get_model(backbone_name, num_classes, dropout_rate):
    if backbone_name == "ResNet50_CAP":
        return CAPResNet(num_classes=num_classes, drop=dropout_rate)
    elif backbone_name == "EffNet_SE":
        return SEEffNet(num_classes=num_classes, drop=dropout_rate)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

# Objective function
def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    backbone_name = trial.suggest_categorical('backbone', ['ResNet50_CAP', 'EffNet_SE'])
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    criterion_name = trial.suggest_categorical('criterion', ['CrossEntropy', 'LabelSmoothing', 'Focal'])

    # Data
    train_loader, val_loader, test_loader, num_classes, class_names = get_loaders(img_size=224, batch_size=batch_size, annot='variant')

    # Model
    model = get_model(backbone_name, num_classes, dropout_rate)
    model.to(device)

    # Optimizer
    optimizer_cls = getattr(optim, optimizer_name)
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Criterion 
    if criterion_name == 'CrossEntropy':
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif criterion_name == 'LabelSmoothing':
        smoothing = trial.suggest_float('smoothing', 0.05, 0.2)
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif criterion_name == 'Focal':
        gamma = trial.suggest_float('gamma', 1.0, 3.0)
        criterion = FocalLoss(gamma=gamma)

    scaler = GradScaler('cuda')

    # Training loop
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print("✅ Model improved. Saving...")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s).")
        if epochs_without_improvement >= patience:
            print("⏹️ Early stopping triggered.")
            break

    return val_acc

