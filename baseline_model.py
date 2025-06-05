import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
from config import Config, ModelType

class BaselineModel: #Full Fine Tuning
    def __init__(self, num_classes, config: Config, device=None, lr=0.001):
        self.cfg = config
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
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

            print(f"[Baseline] Epoch {epoch+1}: Loss = {running_loss/len(dataloader):.4f}")

            # Checkpoint path
            ckpt_path = self.cfg.checkpoint_path(epoch+1, ModelType.Baseline)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    def evaluate_model(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"[Baseline] Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def expand_classifier(self, num_new_classes):
        """
        Poszerza warstwę klasyfikującą o nowe klasy, zachowując wagi.
        """
        in_features = self.model.fc.in_features
        num_old_classes = self.model.fc.out_features
        num_total = num_old_classes + num_new_classes

        old_weights = self.model.fc.weight.data
        old_bias = self.model.fc.bias.data

        new_fc = nn.Linear(in_features, num_total)
        with torch.no_grad():
            new_fc.weight[:num_old_classes] = old_weights
            new_fc.bias[:num_old_classes] = old_bias

        self.model.fc = new_fc.to(self.device)
        self.num_classes = num_total

        print(f"🔧 Expanded classifier: {num_old_classes} → {num_total} classes")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
        print(f"💾 Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        num_classes = checkpoint['num_classes']
        self.num_classes = num_classes

        # Rekonstruuj model z odpowiednią liczbą klas
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = self.model.to(self.device)

        print(f"✅ Model loaded from {path}, classes: {num_classes}")


