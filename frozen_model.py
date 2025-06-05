import torch.nn as nn
from baseline_model import BaselineModel
from config import ModelType
import os
import torch.optim as optim
import torch


class FrozenModel(BaselineModel):
    def __init__(self, num_classes, config, device=None, lr=0.001):
        super().__init__(num_classes, config, device=device, lr=lr)

        # Zamrożenie wszystkich warstw oprócz fc
        for param in self.model.parameters():
            param.requires_grad = False

        # Odmrożenie warstwy klasyfikującej
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.model = self.model.to(self.device)

    def train_model(self, dataloader, num_epochs=10):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"[Frozen] Epoch {epoch+1}: Loss = {running_loss/len(dataloader):.4f}")

            # Zapisz checkpoint
            ckpt_path = self.cfg.checkpoint_path(self.num_classes, ModelType.Frozen)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"📦 Checkpoint saved: {ckpt_path}")
