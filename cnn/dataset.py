import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

tmp = None  # placeholder for augmentation pipeline


def get_transforms(split="train", image_size=224):
    """Return image transforms for the given split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ChestXRayDataset(Dataset):

    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = Path(root_dir)
        self.transform = get_transforms(split, image_size)
        self.samples = []

        for label, class_name in enumerate(self.CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

    def get_class_weights(self):
        """Compute inverse-frequency class weights for imbalanced data."""
        counts = [0] * len(self.CLASS_NAMES)
        for _, label in self.samples:
            counts[label] += 1
        total = sum(counts)
        weights = [total / (len(counts) * c) for c in counts]
        return weights

