import torch
from torch import nn
import torch.nn.functional as F
from .block import block

class ResNet(nn.Module):
    def __init__(self, n, num_classes, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts
        
        # Input
        self.convIn = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnIn   = nn.BatchNorm2d(16, track_running_stats=True)
        self.relu   = nn.ReLU()
        
        # Stack1
        self.stack1 = nn.ModuleList([block(16, subsample=False) for _ in range(n)])

        # Stack2
        self.stack2a = block(32, subsample=True)
        self.stack2b = nn.ModuleList([block(32, subsample=False) for _ in range(n-1)])

        # Stack3
        self.stack3a = block(64, subsample=True)
        self.stack3b = nn.ModuleList([block(64, subsample=False) for _ in range(n-1)])

        # Output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initilise weights in fully connected layer 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()      

    def forward(self, x):     
        z = self.convIn(x)
        z = self.bnIn(z)
        z = self.relu(z)

        for l in self.stack1: z = l(z, shortcuts=self.shortcuts)
        z = self.stack2a(z, shortcuts=self.shortcuts)
        for l in self.stack2b: z = l(z, shortcuts=self.shortcuts)
        z = self.stack3a(z, shortcuts=self.shortcuts)
        for l in self.stack3b: z = l(z, shortcuts=self.shortcuts)

        z = self.avgpool(z)
        z = z.view(z.size(0), -1)

        return z  
