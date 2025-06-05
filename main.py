import logging
import torch
import os
import copy
import pandas as pd
import time
from torch.utils.data import DataLoader
from config import Config, ModelType
from baseline_model import BaselineModel
from frozen_model import FrozenModel
from lwf_model import LwFModel
from ewc_model import EWCModel, EWC
from rehearsal_model import RehearsalModel
from lego_dataset import LegoDataset


def get_logger():
    logger = logging.getLogger('default')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s : %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.info("Logger created")
    return logger


def folder_size_mb(folder):
    return round(sum(
        os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(folder) for f in filenames if f.endswith(('.jpg', '.png', '.jpeg'))
    ) / 1e6, 2)


def run_experiment_add_classes(model_class, model_type, cfg, device, data_root, source, results):
    print(f"\n\u2B06\ufe0f [ADD CLASSES] {model_type.value}")

    train_10 = LegoDataset(data_root, source=source, num_classes=10, split="train")
    test_10 = LegoDataset(data_root, source=source, num_classes=10, split="test")
    train_loader_10 = DataLoader(train_10, batch_size=32, shuffle=True)
    test_loader_10 = DataLoader(test_10, batch_size=32)

    train_20 = LegoDataset(data_root, source=source, num_classes=20, split="train")
    test_20 = LegoDataset(data_root, source=source, num_classes=20, split="test")
    train_loader_20 = DataLoader(train_20, batch_size=32, shuffle=True)
    test_loader_20 = DataLoader(test_20, batch_size=32)

    model = model_class(num_classes=10, config=cfg, device=device)
    model.train_model(train_loader_10, num_epochs=5)
    acc_0 = model.evaluate_model(test_loader_10)

    if model_type == ModelType.Rehearsal:
        model.store_rehearsal_data(train_10)

    model.expand_classifier(num_new_classes=10)
    acc_base_before = model.evaluate_model(test_loader_10)

    start_time = time.time()
    if model_type == ModelType.LwF:
        teacher_model = copy.deepcopy(model.model)
        model.train_model_lwf(train_loader_20, previous_model=teacher_model, num_epochs=5)
    elif model_type == ModelType.EWC:
        ewc = EWC(model.model, train_loader_10, device)
        model.train_model_ewc(train_loader_20, ewc, ewc_lambda=1000, num_epochs=5)
    elif model_type == ModelType.Rehearsal:
        model.train_model_with_rehearsal(train_20, num_epochs=5)
    else:
        model.train_model(train_loader_20, num_epochs=5)
    train_time_stage2 = time.time() - start_time

    acc_1 = model.evaluate_model(test_loader_10)
    acc_novel = model.evaluate_model(test_loader_20)

    forget = acc_base_before - acc_1
    acc_2 = (acc_1 + acc_novel) / 2

    model_path = cfg.checkpoint_path(20, model_type)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.model.state_dict(), model_path)
    model_size_MB = round(os.path.getsize(model_path) / 1e6, 2)
    data_size_MB = folder_size_mb(train_20.root_dir)

    results.append({
        "Method": model_type.value,  # Nazwa metody uczenia
        "Acc (0)": round(acc_0, 1),  # Accuracy na starym zbiorze (10 klas) po pierwszym treningu
        "Acc (1)": round(acc_1, 1),  # Accuracy na starym zbiorze po douczeniu (20 klas)
        "Base ↓": round(acc_1, 1),  # To samo co Acc(1), ułatwia czytanie jako spadek jakości
        "Novel ↑": round(acc_novel, 1),  # Accuracy na nowych klasach (11–20) po douczeniu
        "Forget ↓": round(forget, 1),  # Różnica między jakością przed i po douczeniu na starym zbiorze
        "Acc (2)": round(acc_2, 1),  # Średnia accuracy między starym i nowym zbiorem
        "Time (s)": round(train_time_stage2, 1),  # Czas treningu etapu 2 (douczenie)
        "Model Size (MB)": model_size_MB,  # Rozmiar modelu po douczeniu (plik .pt)
        "Data Size (MB)": data_size_MB  # Rozmiar danych wykorzystanych do douczania
    })


def run_experiment_add_data(model_class, model_type, cfg, device, data_root, source, results):
    print(f"\n📊 [ADD DATA] {model_type.value}")

    train_small = LegoDataset(data_root, source=source, num_classes=10, split="train", split_ratio=(0.3, 0.15, 0.55))
    train_full = LegoDataset(data_root, source=source, num_classes=10, split="train", split_ratio=(0.7, 0.15, 0.15))
    test_full = LegoDataset(data_root, source=source, num_classes=10, split="test")

    train_loader_small = DataLoader(train_small, batch_size=32, shuffle=True)
    train_loader_full = DataLoader(train_full, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_full, batch_size=32)

    model = model_class(num_classes=10, config=cfg, device=device)
    model.train_model(train_loader_small, num_epochs=5)
    acc_0 = model.evaluate_model(test_loader)

    if model_type == ModelType.Rehearsal:
        model.store_rehearsal_data(train_small)

    start_time = time.time()
    if model_type == ModelType.LwF:
        teacher_model = copy.deepcopy(model.model)
        model.train_model_lwf(train_loader_full, previous_model=teacher_model, num_epochs=5)
    elif model_type == ModelType.EWC:
        ewc = EWC(model.model, train_loader_small, device)
        model.train_model_ewc(train_loader_full, ewc, ewc_lambda=1000, num_epochs=5)
    elif model_type == ModelType.Rehearsal:
        model.train_model_with_rehearsal(train_full, num_epochs=5)
    else:
        model.train_model(train_loader_full, num_epochs=5)
    train_time_stage2 = time.time() - start_time

    acc_1 = model.evaluate_model(test_loader)

    model_path = cfg.checkpoint_path(10, model_type)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.model.state_dict(), model_path)
    model_size_MB = round(os.path.getsize(model_path) / 1e6, 2)
    data_size_MB = folder_size_mb(train_full.root_dir)

    results.append({
        "Method": model_type.value + " + more data",
        "Acc (0)": round(acc_0, 1),
        "Acc (1)": round(acc_1, 1),
        "Base ↓": round(acc_1, 1),
        "Novel ↑": "-",
        "Forget ↓": round(acc_0 - acc_1, 1),
        "Acc (2)": round(acc_1, 1),
        "Time (s)": round(train_time_stage2, 1),
        "Model Size (MB)": model_size_MB,
        "Data Size (MB)": data_size_MB
    })


def main():
    logger = get_logger()
    cfg = Config(os.path.join(os.path.dirname(__file__), "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "C:/Users/katar/source/HITL/data/05 - dataset"
    source = "photos"

    results = []

    for model_class, model_type in [
        (BaselineModel, ModelType.Baseline),
        (FrozenModel, ModelType.Frozen),
        (LwFModel, ModelType.LwF),
        (EWCModel, ModelType.EWC),
        (RehearsalModel, ModelType.Rehearsal)
    ]:
        run_experiment_add_classes(model_class, model_type, cfg, device, data_root, source, results)
        run_experiment_add_data(model_class, model_type, cfg, device, data_root, source, results)
        torch.cuda.empty_cache()


    df = pd.DataFrame(results)
    print("\n📊 Continual Learning Results with Resources:\n", df.to_markdown(index=False))
    df.to_csv("continual_learning_results_with_resources.csv", index=False)


if __name__ == "__main__":
    main()
