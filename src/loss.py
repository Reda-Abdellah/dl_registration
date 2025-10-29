import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import uniform_filter1d  # For MI approximation
import torch.nn as nn


def mutual_information(fixed, warped, patch_size=32):
    """Approximate local MI between fixed and warped (multimodal)."""
    # Simple histogram-based MI; for production, use MONAI's LocalNormalizedCrossCorrelation or ITK
    def local_mi(img1, img2, patch_size):
        # Extract patches, compute MI per patch, average
        # Placeholder: Use NCC as proxy for demo
        ncc = F.normalize(img1) * F.normalize(img2)
        return torch.mean(ncc)
    
    return -local_mi(fixed, warped, patch_size)  # Negative for maximization

class RegistrationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_affine = config['loss']['lambda_affine']
        self.lambda_mi = config['loss']['lambda_mi']
    
    def forward(self, predicted_affine, true_affine, fixed, warped):
        affine_loss = F.mse_loss(predicted_affine, true_affine)
        mi_loss = mutual_information(fixed, warped)
        return self.lambda_affine * affine_loss + self.lambda_mi * mi_loss
