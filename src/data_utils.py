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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def get_raw():

    cache_dir = Path.home() / "Capstone" / "FGVCAircraft"
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets.FGVCAircraft(root = str(cache_dir), download=True)
    ROOT = cache_dir / "fgvc-aircraft-2013b" / "data"
    return ROOT    

ROOT = get_raw()
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

def get_loaders(img_size = 224, batch_size = 32, annot = 'manufacturer'):
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]

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