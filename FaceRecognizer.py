import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from FRDataset import FRDataset
from collections import OrderedDict

class FaceRecognizer:
    def __init__(self, csv_path, resnet_model, img_dir=None, batch_size=32, lr=1e-3, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_path = csv_path
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.dataset = FRDataset(csv_path, img_dir, transform=self.transform)
        if img_dir != None:
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.num_classes = len(self.dataset.labels)

        # Patch fcOut layer in your ResNet
        resnet_model.fcOut = nn.Linear(64, self.num_classes)
        self.model = resnet_model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.load_checkpoint()

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for i, (images, labels) in enumerate(self.dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} done. Average loss: {avg_loss:.4f}\n")

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch+1, avg_loss)


    def predict(self, img_path):
        self.model.eval()
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            pred = torch.argmax(outputs, dim=1).item()
            label = self.dataset.idx2label[pred]
            return label

    def save_checkpoint(self, epoch, loss, path='checkpoints/face_recognizer.pth'):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path='checkpoints/face_recognizer.pth'):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']

            # Check if model is wrapped with DataParallel
            model_is_parallel = isinstance(self.model, nn.DataParallel)
            ckpt_is_parallel = list(state_dict.keys())[0].startswith('module.')

            # Fix mismatch between model and checkpoint
            if model_is_parallel and not ckpt_is_parallel:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[f"module.{k}"] = v
                state_dict = new_state_dict

            elif not model_is_parallel and ckpt_is_parallel:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace("module.", "")] = v
                state_dict = new_state_dict

            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Loaded checkpoint from {path}")
            return True
        else:
            print(f"⚠️ Checkpoint file not found at {path}")
            return False