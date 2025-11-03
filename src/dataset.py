import glob
import json
from typing import Any, Dict, Tuple, Union
import torch
import torchio as tio
from torch.utils.data import Dataset
import numpy as np
import yaml
from pathlib import Path
import SimpleITK as sitk
import torch.nn.functional as F


class RegistrationDatasetCTonly(Dataset):
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['data']['root_dir'])
        self.subjects = glob.glob(str(self.data_dir / "**/image.mha"), recursive=True)#[:2]
        print("subjects len:", len(self.subjects))
        
        self.target_spacing = np.array(self.config['preprocess']['target_spacing'])
        self.crop_size_mm = self.config['preprocess']['crop_size_mm']
        self.patch_size_vox = self.config.get('preprocess', {}).get('patch_size_vox', 64)
        
        # Spacing normalization
        self.spacing_transform = tio.Resample(self.target_spacing)
        
        # Intensity normalization
        self.intensity_normalization = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
        ])
        
        # Load subjects with landmarks
        self.pairs = []
        for ct_path in self.subjects:
            ct_path = Path(ct_path)
            subj_dir = ct_path.parent
            landmark_path = subj_dir / "cochlea-estimated.json"
            
            if ct_path.exists() and landmark_path.exists():
                with open(landmark_path, 'r') as f:
                    landmarks = json.load(f)
                    center_mm = np.array(landmarks.get('center', [0, 0, 0]))
                
                self.pairs.append({
                    'ct_path': ct_path,
                    'landmark_center_mm': center_mm,
                    'subject_dir': subj_dir
                })
        
        if not self.pairs:
            raise ValueError("No valid CT + landmark pairs found")
        
        print(f"Loaded {len(self.pairs)} CT-only subjects with landmarks")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _load_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Load medical image and return tensor and affine matrix."""
        img = sitk.ReadImage(str(image_path))
        ct_array = sitk.GetArrayFromImage(img)  # (D, H, W)
        ct_array = np.transpose(ct_array, (2, 1, 0))  # (W, H, D) -> (H, W, D)
        
        # Get affine from ITK/SimpleITK
        spacing = np.array(img.GetSpacing())
        origin = np.array(img.GetOrigin())
        direction = np.array(img.GetDirection()).reshape(3, 3)
        
        # Build 4x4 affine matrix
        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin
        
        ct_tensor = torch.from_numpy(ct_array.astype(np.float32))
        return ct_tensor, affine
    
    def _build_affine_matrix_3x4(self, rotation_angles: np.ndarray, 
                                   scale_factors: np.ndarray, 
                                   translation: np.ndarray) -> np.ndarray:
        """
        Build 3x4 affine matrix consistent with PyTorch's affine_grid.
        
        Args:
            rotation_angles: [rx, ry, rz] in radians
            scale_factors: [sx, sy, sz] scaling factors
            translation: [tx, ty, tz] in normalized coordinates [-1, 1]
        
        Returns:
            theta: 3x4 affine matrix for PyTorch
        """
        rx, ry, rz = rotation_angles
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        rot_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        rot_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Composite rotation: Rz * Ry * Rx
        rotation_matrix = rot_z @ rot_y @ rot_x
        
        # Scale matrix
        scale_matrix = np.diag(scale_factors)
        
        # Combined transformation matrix
        transform_matrix = rotation_matrix @ scale_matrix
        
        # Build 3x4 theta matrix for PyTorch
        theta = np.zeros((3, 4), dtype=np.float32)
        theta[:3, :3] = transform_matrix
        theta[:3, 3] = translation
        
        return theta
    
    def _generate_random_affine_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random affine transformation parameters for cropped patch.
        
        Returns:
            forward_theta: transformation to apply to fixed patch (3x4)
            inverse_theta: inverse transformation (3x4)
        """
        # Rotation: small angles (±15 degrees)
        max_rotation = np.deg2rad(15)
        rotation_angles = np.random.uniform(-max_rotation, max_rotation, 3)
        
        # Scale: ±10% variation
        scale_factors = np.random.uniform(0.9, 1.1, 3)
        
        # Translation: ±0.15 in normalized coordinates [-1, 1]
        max_translation = 0.15
        translation = np.random.uniform(-max_translation, max_translation, 3)
        
        # Forward transformation
        forward_theta = self._build_affine_matrix_3x4(rotation_angles, scale_factors, translation)
        
        # Compute inverse transformation
        # For 3x4 matrix: [R | t], inverse is [R^-1 | -R^-1 @ t]
        R_forward = forward_theta[:3, :3]
        t_forward = forward_theta[:3, 3]
        
        R_inverse = np.linalg.inv(R_forward)
        t_inverse = -R_inverse @ t_forward
        
        inverse_theta = np.zeros((3, 4), dtype=np.float32)
        inverse_theta[:3, :3] = R_inverse
        inverse_theta[:3, 3] = t_inverse
        
        return forward_theta, inverse_theta
    
    def _apply_affine_to_image(self, image: torch.Tensor, 
                               affine_matrix: np.ndarray) -> torch.Tensor:
        """
        Apply affine transformation to image using PyTorch's grid_sample.
        
        Args:
            image: (H, W, D) tensor
            affine_matrix: 3x4 PyTorch affine matrix
        
        Returns:
            warped: (H, W, D) transformed image
        """
        # Add batch and channel dimensions
        image_batch = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        
        # Convert numpy affine to torch tensor
        theta_torch = torch.from_numpy(affine_matrix).unsqueeze(0)  # (1, 3, 4)
        
        # Generate sampling grid
        grid = F.affine_grid(theta_torch, image_batch.size(), align_corners=False)
        
        # Apply grid_sample with proper interpolation
        warped_batch = F.grid_sample(
            image_batch, grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )
        
        # Remove batch and channel dimensions
        warped = warped_batch.squeeze(0).squeeze(0)
        
        return warped
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.pairs[idx]
        ct_path = data['ct_path']
        landmark_center_mm = data['landmark_center_mm']
        
        # Load image
        ct_tensor, affine = self._load_image(ct_path)
        
        # Create TorchIO image
        ct_tio = tio.ScalarImage(tensor=ct_tensor.unsqueeze(0), affine=affine)
        
        # Normalize spacing to 1mm isotropic
        ct_tio = self.spacing_transform(ct_tio)
        
        # Get image shape and convert landmark to voxel coordinates
        image_shape = np.array(ct_tio.shape[1:])  # (H, W, D)
        affine_norm = ct_tio.affine
        
        # Convert landmark from mm to voxel coordinates
        landmark_homo = np.append(landmark_center_mm, 1)
        landmark_center_voxel = np.linalg.inv(affine_norm) @ landmark_homo
        landmark_center_voxel = landmark_center_voxel[:3].astype(int)
        
        # Ensure landmark is within bounds
        landmark_center_voxel = np.clip(
            landmark_center_voxel, 
            [0, 0, 0], 
            image_shape - 1
        )
        
        # Intensity normalization
        ct_tio = self.intensity_normalization(ct_tio)
        fixed_data = ct_tio.data[0]  # (H, W, D) as torch tensor
        
        # Crop around landmark BEFORE applying transformation
        current_spacing = ct_tio.spacing
        
        fixed_cropped = self.crop_volume(
            fixed_data.numpy(),
            landmark_center_voxel,
            patch_size=self.patch_size_vox,
            voxel_size=current_spacing
        )
        
        # Convert cropped fixed to torch tensor
        fixed_cropped_tensor = torch.from_numpy(fixed_cropped).float()
        
        # Generate random affine transformation for the cropped patch size
        forward_affine, inverse_affine = self._generate_random_affine_params()
        
        # Apply forward transformation to create moving image from fixed cropped
        moving_cropped_tensor = self._apply_affine_to_image(fixed_cropped_tensor, forward_affine)
        
        # Prepare output tensors with correct shapes
        # fixed: (1, H, W, D)
        fixed_tensor = fixed_cropped_tensor.unsqueeze(0).float()
        
        # moving: (1, H, W, D)
        moving_tensor = moving_cropped_tensor.unsqueeze(0).float()
        
        # input: (2, H, W, D) - concatenate fixed and moving
        input_tensor = torch.stack([fixed_cropped_tensor, moving_cropped_tensor], dim=0).float()
        
        # Convert affine matrices to tensors
        forward_affine_tensor = torch.from_numpy(forward_affine).float()
        inverse_affine_tensor = torch.from_numpy(inverse_affine).float()
        
        return {
            'input': input_tensor,  # (2, patch_size, patch_size, patch_size)
            'fixed': fixed_tensor,  # (1, patch_size, patch_size, patch_size)
            'moving': moving_tensor,  # (1, patch_size, patch_size, patch_size)
            'forward_affine': forward_affine_tensor,  # (3, 4)
            'inverse_affine': inverse_affine_tensor,  # (3, 4)
            'landmark_original': landmark_center_voxel.astype(float),
        }
    
    def crop_volume(self, img_data: np.ndarray, centroid: np.ndarray, 
                   size_mm: float = None, patch_size: Union[int, tuple] = None,
                   voxel_size: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
        """Crop volume around centroid."""
        img_data = np.asarray(img_data)
        centroid = np.asarray(centroid)
        
        if img_data.ndim != 3:
            raise ValueError(f"Expected 3D image data, got {img_data.ndim}D")
        
        if len(centroid) != 3:
            raise ValueError(f"Expected 3D centroid, got {len(centroid)}D")
        
        h, w, d = img_data.shape
        
        # Determine crop dimensions
        if patch_size is not None:
            if isinstance(patch_size, int):
                crop_shape = np.array([patch_size, patch_size, patch_size])
            else:
                crop_shape = np.asarray(patch_size)
        elif size_mm is not None:
            spacing_mm = np.mean(voxel_size)
            patch_size_vox = int(size_mm / spacing_mm)
            crop_shape = np.array([patch_size_vox, patch_size_vox, patch_size_vox])
        else:
            raise ValueError("Must specify either size_mm or patch_size")
        
        half_shape = crop_shape // 2
        
        # Clip centroid to valid bounds
        min_bounds = half_shape
        max_bounds = np.array([h, w, d]) - half_shape
        centroid = np.clip(centroid, min_bounds, max_bounds).astype(int)
        
        # Compute crop bounds
        x_start = max(0, centroid[0] - half_shape[0])
        x_end = min(h, centroid[0] + (crop_shape[0] - half_shape[0]))
        y_start = max(0, centroid[1] - half_shape[1])
        y_end = min(w, centroid[1] + (crop_shape[1] - half_shape[1]))
        z_start = max(0, centroid[2] - half_shape[2])
        z_end = min(d, centroid[2] + (crop_shape[2] - half_shape[2]))
        
        cropped = img_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # Pad if necessary
        pad_width = []
        for i, (actual, target) in enumerate(zip(cropped.shape, crop_shape)):
            diff = target - actual
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
        
        if any(before > 0 or after > 0 for before, after in pad_width):
            min_val = np.min(img_data[img_data > 0]) if np.any(img_data > 0) else 0
            cropped = np.pad(cropped, pad_width, mode='constant', constant_values=min_val)
        
        return cropped
