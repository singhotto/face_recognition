from PIL import Image
import torch
import os
from torch.utils.data import Dataset

class FaceDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # List image files
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.img_paths = [os.path.join(self.img_dir, f) for f in self.image_filenames]
        self.label_paths = [os.path.join(self.label_dir, os.path.splitext(f)[0] + ".txt") for f in self.image_filenames]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_filename)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Read YOLO labels (0 x_center y_center width height)
        with open(label_path, "r") as f:
            lines = f.readlines()
            bboxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_center, y_center, w, h = map(float, parts)
                bbox = [1.0, x_center, y_center, w, h]  # Add class score = 1.0
                bboxes.append(bbox)

        if self.transform:
            image = self.transform(image)

        bbox_tensor = torch.tensor(bboxes[0], dtype=torch.float32)

        return image, bbox_tensor
