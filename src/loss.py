import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def mutual_information(fixed, warped, bins=64, normalized=True):
    """
    Compute mutual information between fixed and warped images.
    
    Args:
        fixed: Fixed image tensor (B, C, D, H, W) or (B, 1, D, H, W)
        warped: Warped image tensor (B, C, D, H, W) or (B, 1, D, H, W)
        bins: Number of histogram bins
        normalized: If True, compute normalized MI
        
    Returns:
        MI loss (negative for maximization)
    """
    # Ensure single channel
    if fixed.dim() == 5 and fixed.shape[1] > 1:
        fixed = fixed[:, 0:1]  # Take first channel
    if warped.dim() == 5 and warped.shape[1] > 1:
        warped = warped[:, 0:1]  # Take first channel
    
    # Flatten spatial dimensions: (B, 1, D, H, W) -> (B, D*H*W)
    fixed_flat = fixed.reshape(fixed.shape[0], -1)
    warped_flat = warped.reshape(warped.shape[0], -1)
    
    batch_size = fixed_flat.shape[0]
    mi_batch = []
    
    for b in range(batch_size):
        fixed_sample = fixed_flat[b]
        warped_sample = warped_flat[b]
        
        # Normalize to [0, bins-1] for histogram
        fixed_norm = ((fixed_sample - fixed_sample.min()) / 
                     (fixed_sample.max() - fixed_sample.min() + 1e-10) * (bins - 1))
        warped_norm = ((warped_sample - warped_sample.min()) / 
                      (warped_sample.max() - warped_sample.min() + 1e-10) * (bins - 1))
        
        # Convert to long for indexing
        fixed_idx = fixed_norm.long().clamp(0, bins - 1)
        warped_idx = warped_norm.long().clamp(0, bins - 1)
        
        # Build 2D histogram (joint distribution)
        joint_hist = torch.zeros(bins, bins, device=fixed.device)
        indices = torch.stack([fixed_idx, warped_idx], dim=0)
        
        # Count occurrences
        for i in range(indices.shape[1]):
            joint_hist[indices[0, i], indices[1, i]] += 1
        
        # Normalize to probability
        joint_hist = joint_hist / (joint_hist.sum() + 1e-10)
        
        # Marginal distributions
        p_fixed = joint_hist.sum(dim=1)  # Sum over warped
        p_warped = joint_hist.sum(dim=0)  # Sum over fixed
        
        # Compute entropies
        # H(X) = -sum(p(x) * log(p(x)))
        eps = 1e-10
        h_fixed = -torch.sum(p_fixed * torch.log(p_fixed + eps))
        h_warped = -torch.sum(p_warped * torch.log(p_warped + eps))
        h_joint = -torch.sum(joint_hist * torch.log(joint_hist + eps))
        
        # Mutual information: MI(X,Y) = H(X) + H(Y) - H(X,Y)
        mi = h_fixed + h_warped - h_joint
        
        if normalized:
            # Normalized MI: NMI = (H(X) + H(Y)) / H(X,Y)
            # Or alternatively: NMI = 2 * MI / (H(X) + H(Y))
            nmi = 2.0 * mi / (h_fixed + h_warped + eps)
            mi_batch.append(nmi)
        else:
            mi_batch.append(mi)
    
    # Average over batch
    mi_loss = torch.stack(mi_batch).mean()
    
    # Return negative for minimization (we want to maximize MI)
    return -mi_loss


def normalized_cross_correlation(fixed, warped):
    """
    Compute normalized cross-correlation (NCC) as similarity measure.
    More efficient than MI for mono-modal registration.
    
    Args:
        fixed: Fixed image (B, C, D, H, W)
        warped: Warped image (B, C, D, H, W)
        
    Returns:
        NCC loss (negative for maximization)
    """
    # Ensure single channel
    if fixed.shape[1] > 1:
        fixed = fixed[:, 0:1]
    if warped.shape[1] > 1:
        warped = warped[:, 0:1]
    
    # Flatten spatial dimensions
    fixed_flat = fixed.reshape(fixed.shape[0], -1)
    warped_flat = warped.reshape(warped.shape[0], -1)
    
    # Zero-mean normalization
    fixed_mean = fixed_flat.mean(dim=1, keepdim=True)
    warped_mean = warped_flat.mean(dim=1, keepdim=True)
    
    fixed_centered = fixed_flat - fixed_mean
    warped_centered = warped_flat - warped_mean
    
    # Compute NCC
    numerator = (fixed_centered * warped_centered).sum(dim=1)
    denominator = torch.sqrt((fixed_centered ** 2).sum(dim=1) * 
                            (warped_centered ** 2).sum(dim=1)) + 1e-10
    
    ncc = numerator / denominator
    
    # Return negative for minimization (we want to maximize NCC)
    return -ncc.mean()


def local_normalized_cross_correlation(fixed, warped, kernel_size=9):
    """
    Compute local normalized cross-correlation (LNCC).
    Better for capturing local similarities.
    
    Args:
        fixed: Fixed image (B, C, D, H, W)
        warped: Warped image (B, C, D, H, W)
        kernel_size: Size of local window
        
    Returns:
        LNCC loss
    """
    # Ensure single channel
    if fixed.shape[1] > 1:
        fixed = fixed[:, 0:1]
    if warped.shape[1] > 1:
        warped = warped[:, 0:1]
    
    # Create 3D averaging kernel
    ndims = 3
    kernel = torch.ones(1, 1, *([kernel_size] * ndims), 
                       device=fixed.device, dtype=fixed.dtype)
    kernel = kernel / kernel.numel()
    
    # Compute local means
    fixed_mean = F.conv3d(fixed, kernel, padding=kernel_size // 2)
    warped_mean = F.conv3d(warped, kernel, padding=kernel_size // 2)
    
    # Compute local variances and covariance
    fixed_var = F.conv3d(fixed ** 2, kernel, padding=kernel_size // 2) - fixed_mean ** 2
    warped_var = F.conv3d(warped ** 2, kernel, padding=kernel_size // 2) - warped_mean ** 2
    covar = F.conv3d(fixed * warped, kernel, padding=kernel_size // 2) - fixed_mean * warped_mean
    
    # Compute LNCC
    eps = 1e-10
    lncc = (covar ** 2) / (fixed_var * warped_var + eps)
    
    # Return negative mean (we want to maximize LNCC)
    return -lncc.mean()


class RegistrationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_affine = config['loss']['lambda_affine']
        self.lambda_similarity = config['loss'].get('lambda_mi', 
                                                    config['loss'].get('lambda_similarity', 1.0))
        self.similarity_metric = config['loss'].get('similarity_metric', 'ncc')  # 'mi', 'ncc', or 'lncc'
        
        print(f"Using similarity metric: {self.similarity_metric}")
    
    def forward(self, predicted_affine, true_affine, fixed, warped):
        """
        Compute registration loss.
        
        Args:
            predicted_affine: Predicted affine parameters (B, 12)
            true_affine: Ground truth affine parameters (B, 12)
            fixed: Fixed image (B, C, D, H, W)
            warped: Warped/moved image (B, C, D, H, W)
            
        Returns:
            Total loss
        """
        # Affine parameter loss (supervised)
        affine_loss = F.mse_loss(predicted_affine, true_affine)
        
        # Image similarity loss (unsupervised)
        if self.similarity_metric == 'mi':
            similarity_loss = mutual_information(fixed, warped)
        elif self.similarity_metric == 'lncc':
            similarity_loss = local_normalized_cross_correlation(fixed, warped)
        else:  # 'ncc' (default)
            similarity_loss = normalized_cross_correlation(fixed, warped)
        
        # Total loss
        total_loss = self.lambda_affine * affine_loss + self.lambda_similarity * similarity_loss
        
        return total_loss, {
            'affine_loss': affine_loss.item(),
            'similarity_loss': similarity_loss.item(),
            'total_loss': total_loss.item()
        }


# For backward compatibility
def mutual_information_legacy(fixed, warped, patch_size=32):
    """Legacy function - use normalized_cross_correlation instead for efficiency."""
    return normalized_cross_correlation(fixed, warped)
