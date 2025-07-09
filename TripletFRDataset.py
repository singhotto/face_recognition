import os
import random
from torch.utils.data import Dataset
from PIL import Image

class TripletFRDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        # Map: label_name -> list of image paths
        self.label_to_images = {}

        for fname in os.listdir(folder_path):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                label_name = '_'.join(fname.split('_')[:-1])
                path = os.path.join(folder_path, fname)

                if label_name not in self.label_to_images:
                    self.label_to_images[label_name] = []

                self.label_to_images[label_name].append(path)

        # Filter out classes with less than 2 images
        self.label_to_images = {k: v for k, v in self.label_to_images.items() if len(v) >= 2}

        self.labels = list(self.label_to_images.keys())

        # Build list of all valid anchor-positive pairs (slow to index, but safe)
        self.pairs = []
        for label in self.labels:
            imgs = self.label_to_images[label]
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    self.pairs.append((imgs[i], imgs[j], label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_path, positive_path, label = self.pairs[idx]

        # Pick a negative class different from the anchor's label
        negative_label = random.choice([l for l in self.labels if l != label])
        negative_path = random.choice(self.label_to_images[negative_label])

        # Load images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
