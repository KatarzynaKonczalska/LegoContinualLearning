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
        # Zapisz kopię parametrów po pierwszym etapie (tylko te, które istniały i były trenowalne)
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
        loss = 0.0
        for n, p in model.named_parameters():
            if n not in self.params:
                # nowy parametr (np. poszerzony classifier) – nie karzemy
                continue
            # Pomiń BN i bias jeśli ustawione w konfiguracji (często wyłączane z EWC)
            if hasattr(model, 'cfg') and model.cfg.ewc_exclude_bn_bias:
                if '.bias' in n or 'bn' in n.lower():
                    continue
            # Zabezpieczenie na wypadek zmiany kształtu (np. fc.weight po rozszerzeniu)
            if p.shape != self.params[n].shape:
                # dopasuj wspólną część (stare klasy) – zakładając, że stare wagi są na początkowych indeksach
                # Przykład: fc.weight [num_new, in] vs [num_old, in] – weź fragment [:num_old]
                with torch.no_grad():
                    target_param = self.params[n]
                    fisher = self.fisher[n]
                # Wyznacz wspólny prefix kształtu
                common_shape = tuple(min(a, b) for a, b in zip(p.shape, target_param.shape))
                index = tuple(slice(0, s) for s in common_shape)
                diff = (p[index] - target_param[index]) ** 2
                loss += torch.sum(fisher[index] * diff)
            else:
                loss += torch.sum(self.fisher[n] * (p - self.params[n]) ** 2)
        return loss

class EWCModel(BaselineModel):
    def train_model_ewc(self, dataloader, ewc: EWC, ewc_lambda=1000, num_epochs=5):
        param_groups = self.get_param_groups(
            lr_backbone=self.lr * self.cfg.lr_backbone_mult,
            lr_head=self.lr * self.cfg.lr_head_mult,
        )
        optimizer = optim.Adam(param_groups, lr=self.lr, weight_decay=self.cfg.weight_decay)
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
                if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                optimizer.step()
                total_loss += total.item()

            print(f"[EWC] Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
