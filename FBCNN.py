import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceBoxCNN(nn.Module):
    def __init__(self):
        super(FaceBoxCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 5)  # class_prob, x_center, y_center, width, height

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 128, 128]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 64, 64]
        x = self.pool(F.relu(self.conv3(x)))  # -> [B, 64, 32, 32]
        x = x.view(x.size(0), -1)             # -> [B, 65536]
        x = F.relu(self.fc1(x))               # -> [B, 128]
        x = torch.sigmoid(self.fc2(x))        # -> [B, 5]
        return x

