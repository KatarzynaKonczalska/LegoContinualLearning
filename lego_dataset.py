import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class LegoDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 source="photos", 
                 num_classes=10, 
                 split="train", 
                 split_ratio=(0.7, 0.15, 0.15), 
                 transform=None, 
                 seed=42,
                 manual_samples=None,
                 include_classes=None):
        """
        :param root_dir: katalog bazowy (np. '.../05 - dataset')
        :param source: 'photos' lub 'renders'
        :param num_classes: ile klas wczytać (np. 10, 20)
        :param split: 'train', 'val', 'test'
        :param split_ratio: domyślny podział
        :param transform: torchvision transforms
        :param seed: losowość do podziału wewnątrz klas
        :param manual_samples: lista (path, label) do bezpośredniego załadowania
        :param include_classes: opcjonalna lista indeksów klas (w zakresie [0, num_classes)) do włączenia.
                                 Użyteczne do tworzenia zbiorów zawierających tylko nowe klasy (np. 10..19),
                                 przy zachowaniu oryginalnych indeksów etykiet (np. 10..19) zgodnych z klasyfikatorem.
        """
        assert split in {"train", "val", "test"}, "split must be train, val or test"
        if transform is None:
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
                ])
        else:
            self.transform = transform

        self.samples = []
        self.root_dir = os.path.join(root_dir, source)
        self.source = source
        self.split = split
        self.num_classes = num_classes
        self.include_classes = include_classes

        if manual_samples is not None:
            self.samples = manual_samples
            return

        # Stałe klasy na podstawie posortowanej listy (0..num_classes-1)
        all_class_names = sorted(os.listdir(self.root_dir))
        class_names = all_class_names[:num_classes]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        random.seed(seed)

        for cls_name in class_names:
            cls_idx = self.class_to_idx[cls_name]
            if self.include_classes is not None and cls_idx not in self.include_classes:
                continue  # pomiń klasy spoza wybranego zakresu
            class_path = os.path.join(self.root_dir, cls_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            images.sort()
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * split_ratio[0])
            n_val = int(n_total * split_ratio[1])

            if split == "train":
                selected = images[:n_train]
            elif split == "val":
                selected = images[n_train:n_train + n_val]
            else:
                selected = images[n_train + n_val:]

            for fname in selected:
                self.samples.append((os.path.join(class_path, fname), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label
