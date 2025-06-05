import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os

from config import Config, ModelType
from baseline_model import BaselineModel

class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()

        for images, labels in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            outputs = self.model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= len(self.dataloader)

        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += torch.sum(self.fisher[n] * (p - self.params[n])**2)
        return loss

class EWCModel(BaselineModel):
    def train_model_ewc(self, dataloader, ewc: EWC, ewc_lambda=1000, num_epochs=5):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                penalty = ewc.penalty(self.model)
                total = loss + ewc_lambda * penalty
                total.backward()
                optimizer.step()
                total_loss += total.item()

            print(f"[EWC] Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
