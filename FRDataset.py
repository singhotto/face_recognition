import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FRDataset(Dataset):
    def __init__(self, csv_file, img_dir=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.labels = sorted(self.data['label'].unique())
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['id']
        label_name = self.data.iloc[idx]['label']
        label = self.label2idx[label_name]

        img_path = os.path.join(self.img_dir, img_name) if self.img_dir else img_name
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
