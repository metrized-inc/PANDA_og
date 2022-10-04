"""
Contains functions and classes related to data and transformation of data
"""
import os
import argparse
from re import A
import torchvision.transforms as T
import torch
import pytorch_lightning as pl
import numpy as np
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from utils.settings_functions import *
from PIL import Image


# Apply label smoothing to dictionary values
def smooth_labels(label_dict, alpha=0):

    label_dict_smoothed = dict()
    # For each label add smoothing
    for k, v in label_dict.items():
        # Uniform distribution approximation, note that normal class probability is 0.5, not 1/3
        d = torch.tensor([0.5])
        # Compute the weighted average of the labels
        y = v * (1 - alpha) + alpha * d

        label_dict_smoothed[k] = y.float()

    return label_dict_smoothed


# Create a class via dataset
class ClassifierDataset(Dataset):
    def __init__(
        self,
        root_dir,
        label_dict,
        target_samples=None,
        degrad_transforms=None,
        degrad_prob=0.0,
        transform=T.ToTensor(),
        alpha=0,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.label_dict = label_dict
        self.data_raw = self.read_data()  # Contains the filepath, label, folder
        self.alpha = alpha
        # If under/oversampling is sepcified
        if target_samples:
            self.data = self.datadict2list(
                self.sample_dataset(self.data_raw, target_samples)
            )
        else:
            self.data = self.data_raw
        self.degrad_transforms = (
            degrad_transforms  # List transforms which degrade the output to zero label
        )
        self.degrad_prob = degrad_prob
        self.transform = transform

    def read_data(self):
        data = []
        for root_dir, _, files in os.walk(self.root_dir):
            for f in files:
                filepath = os.path.join(root_dir, f)
                folder = os.path.basename(os.path.dirname(filepath))
                # Check labels and append
                if folder in self.label_dict.keys():
                    label = self.label_dict[folder]
                    data.append([filepath, label, folder])
        return data

    def sample_dataset(self, data, target_samples):

        # Separate the data into classes
        classes = target_samples.keys()
        data_class = {}
        # data_class is a dictionary holding lists of classes
        for c in classes:
            data_class[c] = [x for x in data if x[2] == c]
        data_sampled = {}

        # For each class in the dictionary
        for c in classes:
            class_samples = data_class[c]
            n = len(class_samples)

            if n > 0:
                if n > target_samples[c]:
                    # Take a random subset
                    samples = random.choices(class_samples, k=target_samples[c])
                    data_sampled[c] = samples
                elif n < target_samples[c]:
                    # Combine original full set + random subset (with replacement)
                    samples = random.choices(class_samples, k=target_samples[c] - n)
                    data_sampled[c] = class_samples + samples
                else:
                    # Take the original if the two are equal
                    data_sampled[c] = class_samples

        return data_sampled

    # Convert the dictionary with key values of classes into a single list
    def datadict2list(self, data_sampled):
        data = []
        for k, v in data_sampled.items():
            data = data + v
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label, label_name = self.data[idx]  # Read the data in
        img = cv2.imread(filename)  # Read the image and convert
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply specified transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        # Apply degradation transform if specified
        if self.degrad_prob > 0 and self.degrad_transforms is not None:
            p = np.random.rand()
            if p < self.degrad_prob:  # Apply degradation
                tf = random.sample(self.degrad_transforms, 1)[
                    0
                ]  # Sample a transform randomly
                if type(tf) == list:
                    tf = T.Compose(tf)  # Could be a list of transforms, compose
                img = tf(img)
                label = torch.Tensor([0.5 * self.alpha])  # Apply alpha smoothing

        return img, label, label_name, filename


"""
DataModule to be used with the pl.Trainer
"""


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, train_ds=None, val_ds=None, test_ds=None):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        if self.train_ds:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                persistent_workers=True,
            )
        else:
            return None

    def val_dataloader(self):
        if self.val_ds:
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                persistent_workers=True,
            )
        else:
            return None

    def test_dataloader(self):
        if self.test_ds:
            return DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                persistent_workers=True,
            )
        else:
            return None


"""
Creates and returns a MyDataModule object using wandb artifacts
settings: .hjson settings file
run: wandb run that called this function or None if not using a wandb run
"""


def create_datamodule(settings, run=None):

    target_dist = eval(
        settings["target_dist"]
    )  # Get the target over sampled distribution
    train_tf = eval(settings["train_transform"])
    val_tf = eval(settings["val_transform"])
    test_tf = eval(settings["test_transform"])
    alpha = settings["alpha"]
    label_dict = eval(settings["label_dict"])

    train_target_dist = eval(settings["train_target_dist"])

    train_dir = getDataDirectory(settings["train_data"], run)
    train_ds = (
        ClassifierDataset(
            train_dir,
            label_dict,
            train_target_dist,
            transform=train_tf,
            alpha=alpha,
        )
    )

    val_dir = getDataDirectory(settings["val_data"], run)
    val_ds = (
        ClassifierDataset(val_dir, label_dict, transform=val_tf) if val_dir else None
    )

    test_dir = getDataDirectory(settings["test_data"], run)
    test_ds = (
        ClassifierDataset(test_dir, label_dict, transform=test_tf) if test_dir else None
    )

    # Pass all dataset splits into the datamodule

    data_module = MyDataModule(
        settings["batch_size"],
        train_ds,
        val_ds,
        test_ds,
    )

    return data_module
