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
                 seed=42):
        """
        :param root_dir: katalog bazowy (np. '.../05 - dataset')
        :param source: 'photos' lub 'renders'
        :param num_classes: ile klas wczytać (np. 10, 20)
        :param split: 'train', 'val', 'test'
        :param split_ratio: domyślny podział
        :param transform: torchvision transforms
        :param seed: losowość do podziału wewnątrz klas
        """
        assert split in {"train", "val", "test"}, "split must be train, val or test"
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.samples = []
        self.root_dir = os.path.join(root_dir, source)

        # Stałe klasy na podstawie posortowanej listy
        class_names = sorted(os.listdir(self.root_dir))[:num_classes]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        random.seed(seed)

        for cls_name in class_names:
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
                self.samples.append((os.path.join(class_path, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label
