import json
from typing import Any, Dict, Tuple
import torch
import torchio as tio
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import yaml
from pathlib import Path
import SimpleITK as sitk


import torch
from torch.utils.data import DataLoader
from scipy.ndimage import shift  # For random offset in patches
import random


class RegistrationDataset(Dataset):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['data']['root_dir'])
        self.subjects = self.config['data']['subjects']
        self.transforms = self._build_transforms()
        
        # Load all subjects
        self.pairs = []
        for subj in self.subjects:
            ct_path = self.data_dir / f"{subj}_ct.nii.gz"
            t2_path = self.data_dir / f"{subj}_t2.nii.gz"
            if ct_path.exists() and t2_path.exists():
                self.pairs.append((ct_path, t2_path))
        
        # Preprocessing: Normalize and resample (optional)
        self.preprocess = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),  # Normalize to [0,1]
            tio.ZNormalization(),  # Zero mean, unit variance
        ])
    
    def _build_transforms(self):
        """Build random affine transform using TorchIO."""
        return tio.RandomAffine(
            degrees=self.config['transforms']['degrees'],
            translation=self.config['transforms']['translate'],
            scales=self.config['transforms']['scale'],
            shears=self.config['transforms']['shear'],
            image_interpolation='linear',
            default_pad_value=0
        )
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        ct_path, t2_path = self.pairs[idx]
        
        # Load volumes
        ct_img = nib.load(ct_path).get_fdata()
        t2_img = nib.load(t2_path).get_fdata()
        
        # Stack as 4D (1, H, W, D) for TorchIO; concatenate channels
        ct_t2 = np.stack([ct_img, t2_img], axis=0)  # Shape: (2, H, W, D)
        subject = tio.Subject(ct_t2=tio.ScalarImage(tensor=ct_t2))
        
        # Apply random affine transformation
        transformed = self.transforms(subject)
        moving = transformed.ct_t2.data[0]  # Shape: (H, W, D)
        
        # Get the applied affine matrix (TorchIO stores it)
        applied_affine = self.transforms.get_params()  # Actually, TorchIO RandomAffine doesn't expose matrix directly; compute inverse
        
        # To get true affine: Use SimpleITK or compute from params (simplified here; implement matrix extraction)
        # For demo: Assume we extract 3x4 affine matrix from transform params
        # true_affine = self._params_to_matrix(transform_params)  # Implement this
        # inverse_affine = np.linalg.inv(true_affine)  # 12 params flattened
        
        # Placeholder: Generate random affine for demo; replace with actual
        true_inverse = torch.randn(12)  # Flattened 3x4 inverse matrix
        
        # Fixed is original concatenated
        fixed = torch.from_numpy(ct_t2).float()  # (2, H, W, D)
        moving = torch.from_numpy(np.stack([moving[0], moving[1]], axis=0)).float()  # (2, H, W, D)
        
        # Reshape to (C, D, H, W) for PyTorch 3D conv (channels first, then spatial)
        fixed = fixed.permute(1, 2, 3, 0).unsqueeze(0)  # (1, D, H, W, C) -> adjust as needed
        moving = moving.permute(1, 2, 3, 0).unsqueeze(0)
        
        return {'fixed': fixed, 'moving': moving, 'inverse_affine': true_inverse}


class RegistrationDatasetCTonly(Dataset):
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['data']['root_dir'])
        self.subjects = self.config['data']['subjects']
        self.target_spacing = np.array(self.config['preprocess']['target_spacing'])
        self.crop_size_mm = self.config['preprocess']['crop_size_mm']
        
        # Build spatial transforms (will be applied with custom center)
        self.spatial_transform = self._build_spatial_transform()
        
        # Spacing normalization
        self.spacing_transform = tio.Resample(self.target_spacing)
        
        # Intensity normalization
        self.intensity_transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.ZNormalization(),
        ])
        
        # Load subjects with landmarks
        self.pairs = []
        for subj in self.subjects:
            subj_dir = self.data_dir / subj
            ct_path = subj_dir / f"{subj}_ct.nii.gz"
            landmark_path = subj_dir / "landmark.json"
            
            if ct_path.exists() and landmark_path.exists():
                with open(landmark_path, 'r') as f:
                    landmarks = json.load(f)
                    center_mm = np.array(landmarks.get('center', [0, 0, 0]))  # Physical coords [x,y,z] mm
                
                self.pairs.append({
                    'ct_path': ct_path,
                    'landmark_center_mm': center_mm,
                    'subject_dir': subj_dir
                })
        
        if not self.pairs:
            raise ValueError("No valid CT + landmark pairs found")
        
        print(f"Loaded {len(self.pairs)} CT-only subjects with landmarks")
    
    def _build_spatial_transform(self) -> tio.RandomAffine:
        """Build random affine transform for augmentation."""
        return tio.RandomAffine(
            degrees=self.config['transforms']['degrees'],
            translation=self.config['transforms']['translate'],  # Now in mm
            scales=self.config['transforms']['scale'],
            shears=self.config['transforms']['shear'],
            image_interpolation='linear',
            default_pad_value=0,
            # Note: center parameter will be set dynamically in __getitem__
        )
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _physical_to_voxel(self, physical_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Convert physical coordinates (mm) to voxel indices."""
        # Remove translation component from affine for transformation
        A = affine[:3, :3]
        b = affine[:3, 3]
        voxel = np.linalg.solve(A, physical_coords - b)
        return np.round(voxel).astype(int)
    
    def _voxel_to_physical(self, voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Convert voxel coordinates to physical coordinates (mm)."""
        A = affine[:3, :3]
        b = affine[:3, 3]
        physical = A @ voxel_coords + b
        return physical
    def _apply_transform_with_center(self, image: tio.ScalarImage, center_voxel: np.ndarray, 
                                   transform: tio.RandomAffine) -> Tuple[tio.ScalarImage, np.ndarray, np.ndarray]:
        """
        Apply transform with specific center point and return transformed center.
        Returns: transformed_image, applied_affine_matrix, transformed_center_voxel
        """
        # Get random parameters from transform
        params = transform._get_params(image)
        
        # Create SimpleITK transform for accurate matrix extraction
        sitk_img = sitk.GetImageFromArray(image.data)
        sitk_img.SetOrigin(image.origin)
        sitk_img.SetSpacing(image.spacing)
        sitk_img.SetDirection(image.orientation)
        
        # Build Euler3DTransform for rotation/translation around center
        center_physical = self._voxel_to_physical(center_voxel, image.affine)
        euler_transform = sitk.Euler3DTransform(center_physical)
        
        # Apply random rotation
        if 'degrees' in params and params['degrees'][0] != 0:
            rot_deg = np.random.uniform(*params['degrees'])
            euler_transform.SetRotation([0, 0, np.radians(rot_deg)])  # Around Z for simplicity; extend for 3D
        
        # Apply translation in mm
        if 'translation' in params:
            trans_mm = np.random.uniform(-np.array(params['translation']), np.array(params['translation']))
            euler_transform.Translate(trans_mm)
        
        # Apply scaling
        if 'scales' in params:
            scale = np.random.uniform(*params['scales'])
            euler_transform.SetScale([scale, scale, scale])
        
        # Get transformation matrix
        applied_matrix = euler_transform.GetMatrix()
        applied_matrix = np.eye(4)
        applied_matrix[:3, :3] = applied_matrix[:3, :3] @ euler_transform.GetMatrix()
        applied_matrix[:3, 3] += euler_transform.GetTranslation()
        
        # Apply to TorchIO image (resample with custom affine)
        transformed_affine = image.affine @ np.linalg.inv(applied_matrix)
        transformed_image = tio.ScalarImage(tensor=image.data, affine=transformed_affine)
        transformed_image = tio.functional.resample(transformed_image, image.shape)  # Ensure same shape
        
        # Transform the center point
        center_physical_homogeneous = np.append(center_physical, 1)
        transformed_center_physical = applied_matrix @ center_physical_homogeneous
        transformed_center_physical = transformed_center_physical[:3]
        transformed_center_voxel = self._physical_to_voxel(transformed_center_physical, transformed_affine)
        
        return transformed_image, applied_matrix, transformed_center_voxel
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.pairs[idx]
        ct_path = data['ct_path']
        landmark_center_mm = data['landmark_center_mm']
        
        # Load CT image
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata().astype(np.float32)
        affine = ct_nii.affine
        
        # Create TorchIO image
        ct_tio = tio.ScalarImage(tensor=ct_data, affine=affine)
        
        # Step 1: Normalize spacing to 1mm isotropic
        ct_tio = self.spacing_transform(ct_tio)
        ct_data_norm = ct_tio.data  # Now isotropic
        
        # Update affine and convert landmark to new voxel space
        affine_norm = ct_tio.affine
        landmark_center_voxel = self._physical_to_voxel(landmark_center_mm, affine_norm)
        
        # Step 2: Apply spatial augmentation centered on landmark
        transform_instance = tio.RandomAffine(
            degrees=self.config['transforms']['degrees'],
            translation=self.config['transforms']['translate'],
            scales=self.config['transforms']['scale'],
            shears=self.config['transforms']['shear'],
            center=landmark_center_voxel.tolist()  # Set augmentation center
        )
        
        ct_tio_aug, applied_matrix, transformed_center_voxel = self._apply_transform_with_center(
            ct_tio, landmark_center_voxel, transform_instance
        )
        
        # Step 3: Intensity normalization on augmented image
        ct_tio_aug = self.intensity_transform(ct_tio_aug)
        moving_data = ct_tio_aug.data  # Shape: (1, H, W, D) but we'll squeeze
        
        # Step 4: Crop around transformed landmark center
        # Get current voxel size (should be [1,1,1] after resampling)
        current_spacing = ct_tio_aug.spacing
        cropped_moving = self.crop_volume(
            moving_data[0],  # Remove batch dim
            transformed_center_voxel,
            size_mm=self.crop_size_mm,
            voxel_size=current_spacing
        )
        
        # Fixed: Original (non-augmented) cropped around original center
        # Re-crop original after spacing normalization
        fixed_data = self.crop_volume(
            ct_data_norm[0],
            landmark_center_voxel,
            size_mm=self.crop_size_mm,
            voxel_size=current_spacing
        )
        
        # Step 5: Compute true inverse affine parameters
        # Inverse matrix: world_to_voxel * original_affine_inv
        original_affine_inv = np.linalg.inv(affine_norm)
        inverse_matrix = applied_matrix @ original_affine_inv  # Maps transformed back to original
        true_inverse_params = self.matrix_to_affine_params(inverse_matrix)  # 12 params
        
        # Convert to tensors (channels first: C=1 for CT-only)
        fixed = torch.from_numpy(fixed_data).unsqueeze(0).float()  # (1, D, H, W)
        moving = torch.from_numpy(cropped_moving).unsqueeze(0).float()  # (1, D, H, W)
        input_tensor = torch.cat([fixed, moving], dim=0)  # (2, D, H, W) for model input
        true_inverse = torch.tensor(true_inverse_params, dtype=torch.float32)
        
        # For 3D conv: Ensure shape (B, C, D, H, W) - here B=1, C=2
        input_tensor = input_tensor.unsqueeze(0)  # (1, 2, D, H, W)
        
        return {
            'input': input_tensor,
            'fixed': fixed.unsqueeze(0),  # (1, 1, D, H, W)
            'moving': moving.unsqueeze(0),  # (1, 1, D, H, W)
            'inverse_affine': true_inverse,
            'landmark_original': landmark_center_voxel,
            'landmark_transformed': transformed_center_voxel
        }
    
    def matrix_to_affine_params(self, matrix: np.ndarray) -> np.ndarray:
        """Flatten 4x4 affine matrix to 12 parameters (3x4) for STN."""
        # STN expects 3x4 matrix: [R | t; 0 0 0 | 1] -> first 3 rows
        params = matrix[:3, :4].flatten()
        return params
    
    def crop_volume(self, img_data: np.ndarray, centroid: np.ndarray, size_mm: float = 60, 
                    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Crop a cube of size_mm around centroid, handling boundaries with padding.
        Assumes isotropic spacing; extend for anisotropic.
        """
        # Convert mm to voxels (use mean spacing for isotropic)
        spacing_mm = np.mean(voxel_size)
        size_vox = int(size_mm / spacing_mm)
        half_size = size_vox // 2
        
        # Ensure centroid in bounds
        h, w, d = img_data.shape
        centroid = np.clip(centroid, half_size, [h - half_size, w - half_size, d - half_size])
        
        # Compute bounds
        x_start, x_end = int(centroid[0] - half_size), int(centroid[0] + half_size)
        y_start, y_end = int(centroid[1] - half_size), int(centroid[1] + half_size)
        z_start, z_end = int(centroid[2] - half_size), int(centroid[2] + half_size)
        
        # Extract crop
        cropped = img_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # Pad if needed (shouldn't occur with clipping, but for safety)
        target_shape = (size_vox, size_vox, size_vox)
        if cropped.shape != target_shape:
            pad_width = [(0, target_shape[i] - cropped.shape[i]) for i in range(3)]
            cropped = np.pad(cropped, pad_width, mode='constant', constant_values=0)
        
        return cropped