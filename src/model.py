import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalizationNet(nn.Module):
    def __init__(self, in_channels=2, base_features=32):
        super().__init__()
        
        # Multi-scale feature extraction (like ResNet for medical imaging)
        self.conv1 = nn.Conv3d(in_channels, base_features, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm3d(base_features)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(base_features, base_features*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm3d(base_features*2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(base_features*2, base_features*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(base_features*4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = nn.Conv3d(base_features*4, base_features*8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(base_features*8)
        
        # Larger bottleneck for medical images
        self.adaptive_pool = nn.AdaptiveAvgPool3d((6, 6, 6))  # Increased from 4×4×4
        
        # More capacity in fully connected layers
        bottleneck_size = base_features*8 * 6 * 6 * 6  # 256*216 = 55,296
        
        self.fc_loc = nn.Sequential(
            nn.Linear(bottleneck_size, 512),  # Increased capacity
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(True), 
            nn.Dropout(0.2),
            nn.Linear(128, 12)  # 3x4 affine params
        )
        
        # Better initialization for medical images
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Xavier initialization for conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Identity initialization for final layer
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,0], dtype=torch.float))
    
    def forward(self, x):
        # Feature extraction with batch normalization
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling and prediction
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        theta = self.fc_loc(x)
        theta = theta.view(-1, 3, 4)
        return theta

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class MedicalLocalizationNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        
        # Initial conv
        self.conv_init = nn.Conv3d(in_channels, 32, 7, padding=3)
        self.bn_init = nn.BatchNorm3d(32)
        self.pool_init = nn.MaxPool3d(2)
        
        # Residual blocks
        self.res1 = ResidualBlock3D(32, 64)
        self.pool1 = nn.MaxPool3d(2)
        
        self.res2 = ResidualBlock3D(64, 128)
        self.pool2 = nn.MaxPool3d(2)
        
        self.res3 = ResidualBlock3D(128, 256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 12)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # FIX: Identity initialization for final transformation layer
        # This helps the network start from identity transform (no deformation)
        final_layer = self.regressor[-1]
        final_layer.weight.data.zero_()
        # Identity transform in 3x4 flattened format
        final_layer.bias.data.copy_(torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,0], dtype=torch.float))
    
    def forward(self, x):
        x = self.pool_init(F.relu(self.bn_init(self.conv_init(x))))
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.res3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        theta = self.regressor(x)
        return theta.view(-1, 3, 4)


class STN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.localization = MedicalLocalizationNet(in_channels)
    
    def forward(self, x, affine=None):
        moving = x[:, 1:2, :, :, :]  # (B,1,D,H,W)
        batch_size = moving.size(0)
        theta_pred = self.localization(x)  # Always predict (B,3,4)
        
        if affine is not None:
            # Sanity mode: use GT for warping, but return GT for loss comparison too
            theta_gt = affine.view(batch_size, 3, 4)
            theta = theta_gt  # Warp with GT
            # pred_for_loss = theta_gt.view(batch_size, -1)  # Use GT as "pred" for test
        else:
            theta = theta_pred
            # pred_for_loss = theta_pred.view(batch_size, -1)
        
        grid = F.affine_grid(theta, moving.size(), align_corners=False)
        warped = F.grid_sample(moving, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return warped, theta  # Now in test, pred_for_loss == GT flattened

# FIX: Optional - Add a version with residual refinement for better convergence
class STNWithRefinement(nn.Module):
    """
    STN that predicts residual transformation from identity.
    This can help with training stability.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.localization = MedicalLocalizationNet(in_channels)
    
    def forward(self, x, affine=None):
        moving = x[:, 1:2, :, :, :]
        batch_size = moving.size(0)
        
        # Predict RESIDUAL transformation
        theta_residual = self.localization(x)  # (B, 3, 4)
        
        # Add identity to get final transformation
        identity = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,0], 
                               dtype=theta_residual.dtype, 
                               device=theta_residual.device)
        identity = identity.view(1, 3, 4).repeat(batch_size, 1, 1)
        
        theta_pred = theta_residual + identity  # Residual connection
        
        if affine is not None:
            theta = affine.view(batch_size, 3, 4)
        else:
            theta = theta_pred
        
        grid = F.affine_grid(theta, moving.size(), align_corners=False)
        warped = F.grid_sample(
            moving, 
            grid, 
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        return warped, theta_pred.view(batch_size, -1)
