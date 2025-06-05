import logging
import torch
import os
from torch.utils.data import DataLoader
from config import Config, ModelType
from baseline_model import BaselineModel
from frozen_model import FrozenModel
from lego_dataset import LegoDataset
import pandas as pd

def get_logger():
    logger = logging.getLogger('default')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(level=logging.DEBUG)
        formatter =  logging.Formatter('%(levelname)s : %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.info("Logger created")
    return logger

def main():
    logger = get_logger()
    cfg = Config(os.path.join(os.path.dirname(__file__), "config.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "C:/Users/katar/source/HITL/data/05 - dataset"
    source = "photos"  # lub 'renders'

    results = []

    for model_class, model_type in [
        (BaselineModel, ModelType.Baseline),
        (FrozenModel, ModelType.Frozen)
    ]:
        logger.info(f"🚀 Running experiment: {model_type.value}")

        # === Dataset 10 klas
        train_10 = LegoDataset(data_root, source=source, num_classes=10, split="train")
        val_10 = LegoDataset(data_root, source=source, num_classes=10, split="val")
        test_10 = LegoDataset(data_root, source=source, num_classes=10, split="test")

        train_loader_10 = DataLoader(train_10, batch_size=32, shuffle=True)
        test_loader_10 = DataLoader(test_10, batch_size=32)

        # === Dataset 20 klas
        train_20 = LegoDataset(data_root, source=source, num_classes=20, split="train")
        val_20 = LegoDataset(data_root, source=source, num_classes=20, split="val")
        test_20 = LegoDataset(data_root, source=source, num_classes=20, split="test")

        train_loader_20 = DataLoader(train_20, batch_size=32, shuffle=True)
        test_loader_20 = DataLoader(test_20, batch_size=32)

        # === Model: trenuj na 10 klasach
        model = model_class(num_classes=10, config=cfg, device=device)
        model.train_model(train_loader_10, num_epochs=5)
        acc_0 = model.evaluate_model(test_loader_10)

        # === Rozszerz do 20 klas i doucz
        model.expand_classifier(num_new_classes=10)
        acc_base_before = model.evaluate_model(test_loader_10)
        model.train_model(train_loader_20, num_epochs=5)
        acc_1 = model.evaluate_model(test_loader_10)
        acc_novel = model.evaluate_model(test_loader_20)

        forget = acc_base_before - acc_1
        acc_2 = (acc_1 + acc_novel) / 2

        results.append({
            "Method": model_type.value,
            "Acc (0)": round(acc_0, 1),
            "Acc (1)": round(acc_1, 1),
            "Base ↓": round(acc_1, 1),
            "Novel ↑": round(acc_novel, 1),
            "Forget ↓": round(forget, 1),
            "Acc (2)": round(acc_2, 1)
        })

    df = pd.DataFrame(results)
    print("\n📊 Continual Learning Results:\n", df.to_markdown(index=False))

if __name__ == "__main__":
    main()
