import logging
import torch
import os
import copy
import pandas as pd
import random
import numpy as np
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def folder_size_mb(folder):
    return round(sum(
        os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(folder) for f in filenames if f.endswith(('.jpg', '.png', '.jpeg'))
    ) / 1e6, 2)


def run_experiment_add_classes(model_class, model_type, cfg, device, data_root, source, results):
    print(f"\n\u2B06\ufe0f [ADD CLASSES] {model_type.value}")

    train_10 = LegoDataset(data_root, source=source, num_classes=10, split="train")
    test_10 = LegoDataset(data_root, source=source, num_classes=10, split="test")
    train_loader_10 = DataLoader(train_10, batch_size=cfg.batch_size, shuffle=True)
    test_loader_10 = DataLoader(test_10, batch_size=cfg.batch_size)

    # Pełne zbiory 20 klas (do rozmiaru danych), oraz podzbiór nowych klas 10..19 do trenowania w etapie 2
    train_20_all = LegoDataset(data_root, source=source, num_classes=20, split="train")
    test_20_all = LegoDataset(data_root, source=source, num_classes=20, split="test")
    # tylko nowe klasy (indeksy 10..19), etykiety pozostają 10..19 – zgodne z rozszerzonym classifierem
    new_class_indices = list(range(10, 20))
    train_20_new = LegoDataset(data_root, source=source, num_classes=20, split="train", include_classes=new_class_indices)
    test_20_new = LegoDataset(data_root, source=source, num_classes=20, split="test", include_classes=new_class_indices)
    train_loader_20_new = DataLoader(train_20_new, batch_size=cfg.batch_size, shuffle=True)
    test_loader_20_new = DataLoader(test_20_new, batch_size=cfg.batch_size)

    model = model_class(num_classes=10, config=cfg, device=device, lr=cfg.lr_stage1)
    model.train_model(train_loader_10, num_epochs=cfg.train_epochs_stage1)
    acc_0 = model.evaluate_model(test_loader_10)

    if model_type == ModelType.Rehearsal:
        model.store_rehearsal_data(train_10)

    # Snapshot teacher lub EWC przed rozszerzeniem klas
    teacher_model = None
    ewc = None
    if model_type == ModelType.LwF:
        teacher_model = copy.deepcopy(model.model).to(device)
        teacher_model.eval()
    elif model_type == ModelType.EWC:
        ewc = EWC(model.model, train_loader_10, device)

    model.expand_classifier(num_new_classes=10)
    acc_base_before = model.evaluate_model(test_loader_10)

    start_time = time.time()
    # Prepare stage-2 specifics
    model.set_lr(cfg.lr_stage2)
    if cfg.freeze_bn_stage2:
        model.freeze_batchnorm()
    if model_type == ModelType.LwF:
        # Trenuj na nowych klasach z distylacją na starych
        if isinstance(model, LwFModel):
            model.temperature = cfg.lwf_temperature
            model.alpha = cfg.lwf_alpha
        model.train_model_lwf(train_loader_20_new, previous_model=teacher_model, num_epochs=cfg.train_epochs_stage2)
    elif model_type == ModelType.EWC:
        # Trenuj tylko na nowych klasach, karząc odchylenia w parametrach istotnych dla starych
        model.train_model_ewc(train_loader_20_new, ewc, ewc_lambda=cfg.ewc_lambda, num_epochs=cfg.train_epochs_stage2)
    elif model_type == ModelType.Rehearsal:
        # Rehearsal używa nowych danych + bufor starych
        model.train_model_with_rehearsal(train_20_all, num_epochs=cfg.train_epochs_stage2)
    else:
        # Baseline/Frozen: naiwne douczanie TYLKO na nowych klasach (dla fair porównania)
        model.train_model(train_loader_20_new, num_epochs=cfg.train_epochs_stage2)
    train_time_stage2 = time.time() - start_time

    acc_1 = model.evaluate_model(test_loader_10)
    acc_novel = model.evaluate_model(test_loader_20_new)

    # Optional fine-tuning step (post stage-2)
    maybe_fine_tune(model, cfg, device, data_root, source)

    forget = acc_base_before - acc_1
    acc_2 = (acc_1 + acc_novel) / 2

    model_path = cfg.checkpoint_path(20, model_type)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.model.state_dict(), model_path)
    model_size_MB = round(os.path.getsize(model_path) / 1e6, 2)
    data_size_MB = folder_size_mb(train_20_all.root_dir)

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


def maybe_fine_tune(model, cfg, device, data_root, source):
    if not cfg.fine_tune_enable:
        return
    print("\n🔧 Optional fine-tuning enabled")
    # Build datasets for strategies
    old_idx = list(range(0, 10))
    new_idx = list(range(10, 20))
    train_old = LegoDataset(data_root, source=source, num_classes=20, split="train", include_classes=old_idx)
    train_new = LegoDataset(data_root, source=source, num_classes=20, split="train", include_classes=new_idx)

    # Choose dataset per strategy
    if cfg.fine_tune_strategy == 'new_only':
        ft_dataset = train_new
    elif cfg.fine_tune_strategy == 'old_only':
        ft_dataset = train_old
    elif cfg.fine_tune_strategy == 'all':
        # Use all 20 (balanced because dataset provides per-class sampling by split construction)
        ft_dataset = LegoDataset(data_root, source=source, num_classes=20, split="train")
    else:
        # 'mix': mix a fraction of old with all new
        from torch.utils.data import ConcatDataset, Subset
        old_count = int(len(train_old) * cfg.fine_tune_mix_old_frac)
        indices = list(range(len(train_old)))[:max(1, old_count)]
        ft_dataset = ConcatDataset([train_new, Subset(train_old, indices)])

    if cfg.fine_tune_unfreeze_bn:
        # re-enable BN training for fine-tuning if desired
        for m in model.model.modules():
            if hasattr(m, 'track_running_stats'):
                m.train()
            if hasattr(m, 'weight') and hasattr(m, 'bias'):
                for p in m.parameters():
                    p.requires_grad = True

    # Set fine-tune LR
    model.set_lr(cfg.fine_tune_lr)
    dataloader = DataLoader(ft_dataset, batch_size=cfg.batch_size, shuffle=True)
    # Temporarily override weight decay via simple re-train call (train_model uses cfg.weight_decay)
    orig_wd = model.cfg.weight_decay
    model.cfg.config.setdefault('train', {})
    model.cfg.config['train']['weight_decay'] = cfg.fine_tune_weight_decay
    model.train_model(dataloader, num_epochs=cfg.fine_tune_epochs)
    model.cfg.config['train']['weight_decay'] = orig_wd


def run_experiment_add_data(model_class, model_type, cfg, device, data_root, source, results):
    print(f"\n📊 [ADD DATA] {model_type.value}")

    train_small = LegoDataset(data_root, source=source, num_classes=10, split="train", split_ratio=(0.3, 0.15, 0.55))
    train_full = LegoDataset(data_root, source=source, num_classes=10, split="train", split_ratio=(0.7, 0.15, 0.15))
    test_full = LegoDataset(data_root, source=source, num_classes=10, split="test")

    train_loader_small = DataLoader(train_small, batch_size=cfg.batch_size, shuffle=True)
    train_loader_full = DataLoader(train_full, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_full, batch_size=cfg.batch_size)

    model = model_class(num_classes=10, config=cfg, device=device, lr=cfg.lr_stage1)
    model.train_model(train_loader_small, num_epochs=cfg.train_epochs_stage1)
    acc_0 = model.evaluate_model(test_loader)

    if model_type == ModelType.Rehearsal:
        model.store_rehearsal_data(train_small)

    start_time = time.time()
    model.set_lr(cfg.lr_stage2)
    if cfg.freeze_bn_stage2:
        model.freeze_batchnorm()
    if model_type == ModelType.LwF:
        teacher_model = copy.deepcopy(model.model)
        if isinstance(model, LwFModel):
            model.temperature = cfg.lwf_temperature
            model.alpha = cfg.lwf_alpha
        model.train_model_lwf(train_loader_full, previous_model=teacher_model, num_epochs=cfg.train_epochs_stage2)
    elif model_type == ModelType.EWC:
        ewc = EWC(model.model, train_loader_small, device)
        model.train_model_ewc(train_loader_full, ewc, ewc_lambda=cfg.ewc_lambda, num_epochs=cfg.train_epochs_stage2)
    elif model_type == ModelType.Rehearsal:
        model.train_model_with_rehearsal(train_full, num_epochs=cfg.train_epochs_stage2)
    else:
        model.train_model(train_loader_full, num_epochs=cfg.train_epochs_stage2)
    train_time_stage2 = time.time() - start_time

    acc_1 = model.evaluate_model(test_loader)

    # Optional fine-tuning step (post stage-2)
    maybe_fine_tune(model, cfg, device, data_root, source)

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
    set_seed(42)

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
