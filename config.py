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
