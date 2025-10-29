import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalizationNet(nn.Module):
    def __init__(self, in_channels, channels=[8, 10]):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, channels[0], kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(channels[0], channels[1], kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool3d(2)
        
        # Assume input size (e.g., 64x64x64 after pooling -> 16x16x16)
        self.fc_loc = nn.Sequential(
            nn.Linear(channels[1] * 16 * 16 * 16, 32),  # Adjust based on input size
            nn.ReLU(True),
            nn.Linear(32, 12)  # 3x4 affine params
        )
        
        # Initialize to identity
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1,0,0, 0,1,0, 0,0,1, 0,0,0], dtype=torch.float))  # Identity 3x4 flattened
    
    def forward(self, x):
        xs = self.pool1(F.relu(self.conv1(x)))
        xs = self.pool2(F.relu(self.conv2(xs)))
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)  # For 3D affine_grid
        return theta

class STN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.localization = LocalizationNet(in_channels)
    
    def forward(self, moving):
        theta = self.localization(moving)
        # For 3D, use affine_grid with size from moving
        grid = F.affine_grid(theta, moving.size(), align_corners=False)
        warped = F.grid_sample(moving, grid, align_corners=False, mode='bilinear', padding_mode='border')
        return warped, theta.view(theta.size(0), -1)  # Return warped and flattened params
