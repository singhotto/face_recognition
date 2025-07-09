import os
from torch.utils.data import Dataset
from PIL import Image

class FRDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.samples = []

        # Auto-build label map
        self.label_map = {}
        current_label = 0

        for fname in os.listdir(folder_path):
            if fname.endswith(('.jpg', '.png', '.jpeg')):
                label_name = '_'.join(fname.split('_')[:-1])  # e.g., "Akash Deep"
                
                if label_name not in self.label_map:
                    self.label_map[label_name] = current_label
                    current_label += 1

                full_path = os.path.join(folder_path, fname)
                label = self.label_map[label_name]
                self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
