import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
from torch.utils.data import ConcatDataset

from config import Config, ModelType
from baseline_model import BaselineModel

class RehearsalModel(BaselineModel):
    def __init__(self, num_classes, config: Config, device=None, lr=0.001):
        super().__init__(num_classes, config, device, lr)
        self.old_dataset = None

    def store_rehearsal_data(self, dataset, max_samples_per_class=20):
        """
        Przechowuje próbkę danych (np. 20 obrazów na klasę) do późniejszego douczania.
        """
        from collections import defaultdict
        from random import shuffle

        class_images = defaultdict(list)
        for path, label in dataset.samples:
            class_images[label].append((path, label))

        rehearsal_samples = []
        for label, items in class_images.items():
            shuffle(items)
            rehearsal_samples.extend(items[:max_samples_per_class])

        self.old_dataset = dataset.__class__(
            root_dir=dataset.root_dir,  # użyj tego samego typu datasetu
            source=dataset.source,
            num_classes=dataset.num_classes,
            split=dataset.split,
            transform=dataset.transform,
            manual_samples=rehearsal_samples
        )

    def train_model_with_rehearsal(self, new_dataset, num_epochs=5):
        if self.old_dataset is None:
            raise ValueError("Brak danych do rehearsal. Użyj store_rehearsal_data() wcześniej.")

        combined_dataset = ConcatDataset([new_dataset, self.old_dataset])
        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=32, shuffle=True)
        self.train_model(dataloader, num_epochs=num_epochs)
