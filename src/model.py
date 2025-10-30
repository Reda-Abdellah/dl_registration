import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalizationNet(nn.Module):
    def __init__(self, in_channels, channels=[8, 10]):
        super().__init__()
        # Fix: Add out_channels parameter (first element of channels list)
        self.conv1 = nn.Conv3d(in_channels, channels[0], kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(channels[0], channels[1], kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool3d(2)
        
        # Use adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        # Fixed size after adaptive pooling
        self.fc_loc = nn.Sequential(
            nn.Linear(channels[1] * 4 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 12)  # 3x4 affine params
        )
        
        # Initialize to identity transformation
        self.fc_loc[-1].weight.data.zero_()
        # Correct identity matrix for 3x4 affine: [1,0,0,0, 0,1,0,0, 0,0,1,0]
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,0], dtype=torch.float))
    
    def forward(self, x):
        xs = self.pool1(F.relu(self.conv1(x)))
        xs = self.pool2(F.relu(self.conv2(xs)))
        
        # Use adaptive pooling for consistent output size
        xs = self.adaptive_pool(xs)
        xs = xs.view(xs.size(0), -1)
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)  # For 3D affine_grid
        return theta

class STN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.localization = LocalizationNet(in_channels)
    
    def forward(self, x):
        # Extract moving image (second channel)
        moving = x[:, 1:2]  # Shape: (B, 1, D, H, W)
        
        # Predict transformation parameters
        theta = self.localization(x)
        
        # Apply transformation to moving image
        grid = F.affine_grid(theta, moving.size(), align_corners=False)
        warped = F.grid_sample(moving, grid, align_corners=False, mode='bilinear', padding_mode='border')
        
        return warped, theta.view(theta.size(0), -1)  # Return warped and flattened params
