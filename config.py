import json
from pathlib import Path
from enum import Enum


class ModelType(Enum):
    Baseline = "Baseline"
    Frozen = "Frozen"
    LwF = "LwF"
    EWC = "EWC"
    Rehearsal = "Rehearsal"

class DatasetType(Enum):
    Train = "train"
    Val = "val"
    Test = "test"


class Config:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

    @property
    def train_epochs_stage1(self) -> int:
        return int(self.config.get("train", {}).get("epochs_stage1", 10))

    @property
    def train_epochs_stage2(self) -> int:
        return int(self.config.get("train", {}).get("epochs_stage2", 10))

    @property
    def batch_size(self) -> int:
        return int(self.config.get("train", {}).get("batch_size", 32))

    @property
    def lr(self) -> float:
        return float(self.config.get("train", {}).get("lr", 1e-3))

    @property
    def weight_decay(self) -> float:
        return float(self.config.get("train", {}).get("weight_decay", 0.0))

    @property
    def lr_stage1(self) -> float:
        return float(self.config.get("train", {}).get("lr_stage1", self.lr))

    @property
    def lr_stage2(self) -> float:
        return float(self.config.get("train", {}).get("lr_stage2", self.lr))

    @property
    def freeze_bn_stage2(self) -> bool:
        return bool(self.config.get("train", {}).get("freeze_bn_stage2", True))

    @property
    def ewc_lambda(self) -> float:
        return float(self.config.get("ewc", {}).get("lambda", 200.0))

    @property
    def ewc_exclude_bn_bias(self) -> bool:
        return bool(self.config.get("ewc", {}).get("exclude_bn_bias", True))

    @property
    def grad_clip_norm(self) -> float:
        return float(self.config.get("train", {}).get("grad_clip_norm", 1.0))

    @property
    def lr_backbone_mult(self) -> float:
        return float(self.config.get("train", {}).get("lr_backbone_mult", 0.5))

    @property
    def lr_head_mult(self) -> float:
        return float(self.config.get("train", {}).get("lr_head_mult", 1.0))

    @property
    def lwf_alpha(self) -> float:
        return float(self.config.get("lwf", {}).get("alpha", 0.5))

    @property
    def lwf_temperature(self) -> float:
        return float(self.config.get("lwf", {}).get("temperature", 2.0))

    # Fine-tuning settings
    @property
    def fine_tune_enable(self) -> bool:
        return bool(self.config.get("fine_tune", {}).get("enable", False))

    @property
    def fine_tune_epochs(self) -> int:
        return int(self.config.get("fine_tune", {}).get("epochs", 5))

    @property
    def fine_tune_lr(self) -> float:
        return float(self.config.get("fine_tune", {}).get("lr", 1e-4))

    @property
    def fine_tune_weight_decay(self) -> float:
        return float(self.config.get("fine_tune", {}).get("weight_decay", self.weight_decay))

    @property
    def fine_tune_strategy(self) -> str:
        # one of: 'new_only', 'old_only', 'all', 'mix'
        return str(self.config.get("fine_tune", {}).get("strategy", "mix"))

    @property
    def fine_tune_mix_old_frac(self) -> float:
        return float(self.config.get("fine_tune", {}).get("mix_old_frac", 0.3))

    @property
    def fine_tune_unfreeze_bn(self) -> bool:
        return bool(self.config.get("fine_tune", {}).get("unfreeze_bn", True))

    def get_dataset_path(self, num_classes: int, split: DatasetType) -> str:
        key = f"{num_classes}_classes"
        try:
            return self.config["datasets"][key][split.value]
        except KeyError:
            raise ValueError(f"Nie znaleziono ścieżki dla splitu '{split.value}' przy {num_classes} klasach.")

    def checkpoint_path(self, num_classes: int, model_type: ModelType) -> str:
        try:
            base_dir = self.config["checkpoints"]["save_dir"]
            return str(Path(base_dir) / model_type.value / f"{num_classes}_classes" / "model.pt")
        except KeyError as e:
            raise ValueError(f"Błąd w konfiguracji checkpoints: brakujący klucz {e}")
