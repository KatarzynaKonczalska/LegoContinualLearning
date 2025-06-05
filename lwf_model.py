import torch
import torch.nn as nn
import torch.optim as optim
from baseline_model import BaselineModel
from config import ModelType
import os


class LwFModel(BaselineModel):
    def __init__(self, num_classes, config, device=None, lr=0.001, temperature=2.0, alpha=1.0):
        super().__init__(num_classes, config, device=device, lr=lr)
        self.temperature = temperature  # do softmaxu
        self.alpha = alpha              # waga strat distylacyjnych

    def train_model_lwf(self, dataloader, previous_model, num_epochs=5):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_distill = nn.KLDivLoss(reduction='batchmean')

        self.model.train()
        previous_model.eval()
        previous_model.to(self.device)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                # Aktualny model
                outputs = self.model(images)

                # Model referencyjny
                with torch.no_grad():
                    teacher_outputs = previous_model(images)

                # Rozdziel predykcje na klasy stare i nowe
                num_old_classes = teacher_outputs.shape[1]
                student_old = outputs[:, :num_old_classes]
                student_new = outputs

                # 1. klasyfikacja na etykietach (nowe klasy)
                loss_cls = criterion_cls(student_new, labels)

                # 2. distylacja logitów starych klas
                # zastosuj softmax z temperaturą
                T = self.temperature
                distill_loss = criterion_distill(
                    nn.functional.log_softmax(student_old / T, dim=1),
                    nn.functional.softmax(teacher_outputs / T, dim=1)
                ) * (T * T)

                total = self.alpha * distill_loss + (1 - self.alpha) * loss_cls
                total.backward()
                optimizer.step()
                total_loss += total.item()

            print(f"[LwF] Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

            # Checkpoint
            ckpt_path = self.cfg.checkpoint_path(self.num_classes, ModelType.LwF)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")
