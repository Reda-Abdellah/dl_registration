import glob
import json
from typing import Any, Dict, Tuple, Union
import torch
import torchio as tio
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import yaml
from pathlib import Path
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import shift
import random


class RegistrationDatasetCTonly(Dataset):
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['data']['root_dir'])
        self.subjects = glob.glob(str(self.data_dir / "**/image.mha"), recursive=True)#[:10]
        print("subjects len:", len(self.subjects))
        
        self.target_spacing = np.array(self.config['preprocess']['target_spacing'])
        self.crop_size_mm = self.config['preprocess']['crop_size_mm']
        self.patch_size_vox = self.config.get('preprocess', {}).get('patch_size_vox', 64)
        
        # Build spatial transforms
        self.spatial_transform = self._build_spatial_transform()
        
        # Spacing normalization
        self.spacing_transform = tio.Resample(self.target_spacing)
        
        # Intensity normalization
        self.intensity_normalization = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
            # tio.ZNormalization(),
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
    
    def _build_spatial_transform(self) -> tio.RandomAffine:
        """Build random affine transform for augmentation."""
        return tio.RandomAffine(
            degrees=self.config['transforms']['degrees'],
            translation=self.config['transforms']['translate'],
            scales=self.config['transforms']['scale'],
            image_interpolation='linear',
            default_pad_value=0,
        )
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _physical_to_voxel(self, physical_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Convert physical coordinates (mm) to voxel indices."""
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

    def get_transform_center(self, landmark_center_voxel: np.ndarray, image_shape: np.ndarray, 
                       search_radius_mm: float = 5.0, max_attempts: int = 10) -> np.ndarray:
        """Sample a plausible transform center near landmark."""
        landmark_center_voxel = np.asarray(landmark_center_voxel)
        image_shape = np.asarray(image_shape)
        
        if len(image_shape) != 3:
            raise ValueError(f"Expected 3D image_shape, got {len(image_shape)}D: {image_shape}")
        
        search_radius_vox = int(search_radius_mm / np.mean(self.target_spacing))
        
        h, w, d = image_shape
        min_bounds = np.array([search_radius_vox, search_radius_vox, search_radius_vox])
        max_bounds = np.array([h-search_radius_vox, w-search_radius_vox, d-search_radius_vox])
        
        landmark_center_voxel = np.clip(landmark_center_voxel, min_bounds, max_bounds)
        
        attempts = 0
        while attempts < max_attempts:
            offset = np.random.uniform(-search_radius_vox, search_radius_vox, 3)
            candidate_center = landmark_center_voxel + offset
            
            if (np.all(candidate_center >= min_bounds) and 
                np.all(candidate_center < max_bounds)):
                
                distance_vox = np.linalg.norm(offset)
                if distance_vox <= search_radius_vox:
                    return candidate_center.astype(int)
            
            attempts += 1
        
        print(f"Warning: Using landmark center after {max_attempts} attempts.")
        return landmark_center_voxel.astype(int)
    
    def _apply_transform_with_center(self, image: tio.ScalarImage, center_physical: np.ndarray) -> Tuple[tio.ScalarImage, np.ndarray, np.ndarray]:
        """Apply transform by manually sampling parameters."""
        
        # Sample transformation parameters manually
        degrees_range = self.config['transforms']['degrees']  # e.g., [-15, 15]
        translate_range = self.config['transforms']['translate']  # e.g., [0.3, 0.3, 0.3]
        scale_range = self.config['transforms']['scale']  # e.g., [0.9, 1.1]
        
        # Sample random parameters
        rotation_x = np.random.uniform(degrees_range[0], degrees_range[1])
        rotation_y = np.random.uniform(degrees_range[0], degrees_range[1]) 
        rotation_z = np.random.uniform(degrees_range[0], degrees_range[1])
        
        translation_x = np.random.uniform(-translate_range[0], translate_range[0])
        translation_y = np.random.uniform(-translate_range[1], translate_range[1])
        translation_z = np.random.uniform(-translate_range[2], translate_range[2])
        
        scale = np.random.uniform(scale_range[0], scale_range[1])
        # print(rotation_x, rotation_y, rotation_z, translation_x, translation_y, translation_z, scale)
        
        # Convert to transformation matrix
        applied_matrix = self._create_affine_matrix(
            rotation_x, rotation_y, rotation_z,
            translation_x, translation_y, translation_z,
            scale
        )
        
        # Apply transformation using the matrix
        transformed_image = self._apply_matrix_to_image(image, applied_matrix, center_physical)
        
        return transformed_image, applied_matrix

    def _create_affine_matrix(self, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z, scale):
        """Create 4x4 affine transformation matrix from parameters."""
        
        # Convert degrees to radians
        rot_x = np.radians(rot_x)
        rot_y = np.radians(rot_y) 
        rot_z = np.radians(rot_z)
        
        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x)],
            [0, np.sin(rot_x), np.cos(rot_x)]
        ])
        
        Ry = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        
        Rz = np.array([
            [np.cos(rot_z), -np.sin(rot_z), 0],
            [np.sin(rot_z), np.cos(rot_z), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        
        # Scale matrix
        S = np.eye(3) * scale
        
        # Combined rotation and scale
        RS = R @ S
        
        # Create 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = RS
        affine_matrix[:3, 3] = [trans_x, trans_y, trans_z]
        
        return affine_matrix

    def _apply_matrix_to_image(self, image: tio.ScalarImage, transform_matrix: np.ndarray, center_physical: np.ndarray) -> tio.ScalarImage:
        """Apply 4x4 transformation matrix to image using SimpleITK."""
               
        # Convert TorchIO image to SimpleITK
        sitk_image = self._torchio_to_sitk(image)
        
        # Create SimpleITK affine transform from matrix
        affine_transform = sitk.AffineTransform(3)  # 3D affine transform
        
        # SimpleITK expects the transformation matrix in a specific format
        # Extract rotation/scale matrix (3x3) and translation vector (3x1)
        matrix_3x3 = transform_matrix[:3, :3].flatten()  # Flatten to 9 elements
        translation = transform_matrix[:3, 3]
        
        # Set parameters: [matrix elements (9), translation (3)]
        affine_transform.SetMatrix(matrix_3x3.tolist())
        affine_transform.SetTranslation(translation.tolist())

        affine_transform.SetCenter(center_physical)
        
        # Apply transformation with resampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_image)  # Use original image as reference
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(affine_transform)
        
        # Resample the image
        transformed_sitk = resampler.Execute(sitk_image)
        
        # Convert back to TorchIO
        transformed_image = self._sitk_to_torchio(transformed_sitk, image.affine)
        
        return transformed_image

    def _torchio_to_sitk(self, tio_image: tio.ScalarImage) -> sitk.Image:
        """Convert TorchIO image to SimpleITK image."""
        
        # Get numpy array from TorchIO tensor
        # TorchIO format: (1, H, W, D) -> SimpleITK format: (D, H, W)
        numpy_array = tio_image.data.squeeze(0).numpy()  # Remove channel dimension
        numpy_array = np.transpose(numpy_array, (2, 1, 0))  # (H, W, D) -> (D, W, H)
        
        # Create SimpleITK image
        sitk_image = sitk.GetImageFromArray(numpy_array)
        
        # Set spacing and origin from TorchIO affine matrix
        affine = tio_image.affine#.numpy()
        
        # Extract spacing (diagonal elements)
        spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        sitk_image.SetSpacing(spacing.tolist())
        
        # Extract origin (translation part)
        origin = affine[:3, 3]
        sitk_image.SetOrigin(origin.tolist())
        
        # Extract direction (normalized rotation/orientation)
        direction = affine[:3, :3] / spacing
        sitk_image.SetDirection(direction.flatten().tolist())
        
        return sitk_image

    def _sitk_to_torchio(self, sitk_image: sitk.Image, original_affine: torch.Tensor) -> tio.ScalarImage:
        """Convert SimpleITK image back to TorchIO image."""
        
        # Get numpy array from SimpleITK
        numpy_array = sitk.GetArrayFromImage(sitk_image)  # Shape: (D, W, H)
        numpy_array = np.transpose(numpy_array, (2, 1, 0))  # (D, W, H) -> (H, W, D)
        
        # Convert to tensor and add channel dimension
        tensor = torch.from_numpy(numpy_array).float().unsqueeze(0)  # (1, H, W, D)
        
        # Create TorchIO image (keep original affine for consistency)
        tio_image = tio.ScalarImage(tensor=tensor, affine=original_affine)
        
        return tio_image
    
    def _build_affine_from_sitk(self, sitk_img: sitk.Image) -> np.ndarray:
        """Construct 4x4 affine matrix from SimpleITK image metadata."""
        origin = np.array(sitk_img.GetOrigin())
        spacing = np.array(sitk_img.GetSpacing())
        direction_flat = np.array(sitk_img.GetDirection())
        
        if len(direction_flat) != 9:
            raise ValueError(f"Expected 9 direction elements for 3D image, got {len(direction_flat)}")
        
        direction = direction_flat.reshape(3, 3)
        
        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin
        
        return affine
    
    def _load_mha_as_nib_style(self, mha_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load .mha file and return (data, affine)."""
        if not mha_path.exists():
            raise FileNotFoundError(f"MHA file not found: {mha_path}")
        
        try:
            sitk_img = sitk.ReadImage(str(mha_path))
            data_sitk = sitk.GetArrayFromImage(sitk_img)  # (Z, Y, X)
            data = np.transpose(data_sitk, (2, 1, 0))  # (X, Y, Z)
            affine = self._build_affine_from_sitk(sitk_img)
            data = data.astype(np.float32)
            
            return data, affine
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MHA file {mha_path}: {e}")
    
    def _load_nii_as_nib_style(self, nii_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load .nii.gz file and return (data, affine)."""
        if not nii_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {nii_path}")
        
        try:
            ct_nii = nib.load(nii_path)
            data = ct_nii.get_fdata().astype(np.float32)
            affine = ct_nii.affine
            
            return data, affine
            
        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI file {nii_path}: {e}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.pairs[idx]
        ct_path = data['ct_path']
        landmark_center_mm = data['landmark_center_mm']
        
        # Load CT image
        if ct_path.suffix.lower() in ['.mha', '.mhd']:
            ct_data, affine = self._load_mha_as_nib_style(ct_path)
        else:
            ct_data, affine = self._load_nii_as_nib_style(ct_path)
        
        # Add channel dimension for TorchIO: (X, Y, Z) -> (1, X, Y, Z)
        if ct_data.ndim == 3:
            ct_tensor = torch.from_numpy(ct_data).float().unsqueeze(0)
        else:
            raise ValueError(f"Expected 3D data, got {ct_data.ndim}D")
        
        # Create TorchIO image
        ct_tio = tio.ScalarImage(tensor=ct_tensor, affine=affine)
        
        # Normalize spacing to 1mm isotropic
        ct_tio = self.spacing_transform(ct_tio)
        
        # Convert torch.Size to NumPy array
        image_shape = np.array(ct_tio.shape[1:])  # (X, Y, Z) excluding channel
        affine_norm = ct_tio.affine
        
        landmark_center_voxel = self._physical_to_voxel(landmark_center_mm, affine_norm)
        
        # Clip landmark to valid bounds
        min_bounds = np.array([2, 2, 2])
        max_bounds = image_shape - 2
        landmark_center_voxel = np.clip(landmark_center_voxel, min_bounds, max_bounds).astype(int)
        
        # Sample transform center
        # transform_center_voxel = self.get_transform_center(landmark_center_voxel, image_shape)
        
        # Apply spatial augmentation
        # Apply spatial augmentation
        try:
            ct_tio_aug, applied_matrix = self._apply_transform_with_center(ct_tio, landmark_center_mm)
        except Exception as e:
            print(f"Transform error for sample {idx}: {e}. Using identity.")
            ct_tio_aug = ct_tio
            applied_matrix = np.eye(4)
        
        # Intensity normalization
        ct_tio_aug = self.intensity_normalization(ct_tio_aug)
        ct_tio = self.intensity_normalization(ct_tio)
        
        # Extract data tensors (shape: (1, X, Y, Z))
        fixed_data = ct_tio.data[0]  # (X, Y, Z)
        moving_data = ct_tio_aug.data[0]  # (X, Y, Z)
        
        # Crop around centers
        current_spacing = ct_tio_aug.spacing
        
        fixed_cropped = self.crop_volume(
            fixed_data.numpy(),
            landmark_center_voxel,
            patch_size=self.patch_size_vox,
            voxel_size=current_spacing
        )
        
        moving_cropped = self.crop_volume(
            moving_data.numpy(),
            landmark_center_voxel,
            patch_size=self.patch_size_vox,
            voxel_size=current_spacing
        )
        
        # Compute inverse affine parameters
        inverse_matrix = np.linalg.inv(applied_matrix)
        inverse_params  = self.matrix_to_affine_params(inverse_matrix)
        
        # Use FORWARD transformation parameters (not inverse)
        forward_params = self.matrix_to_affine_params(applied_matrix)
        
        # Convert to tensors with correct dimensions
        fixed_tensor = torch.from_numpy(fixed_cropped).float().unsqueeze(0).unsqueeze(0)
        moving_tensor = torch.from_numpy(moving_cropped).float().unsqueeze(0).unsqueeze(0)
        
        # Stack as input: (1, 2, X, Y, Z)
        input_tensor = torch.cat([fixed_tensor, moving_tensor], dim=1)
        
        # Permute to (B, C, D, H, W) = (1, 2, Z, Y, X)
        input_tensor = input_tensor.permute(0, 1, 4, 3, 2)
        fixed_tensor = fixed_tensor.permute(0, 1, 4, 3, 2)
        moving_tensor = moving_tensor.permute(0, 1, 4, 3, 2)
        
        forward_affine = torch.tensor(forward_params, dtype=torch.float32)
        inverse_affine = torch.tensor(inverse_params, dtype=torch.float32)
        
        return {
            'input': input_tensor.squeeze(0),  # (2, D, H, W)
            'fixed': fixed_tensor.squeeze(0),  # (1, D, H, W)
            'moving': moving_tensor.squeeze(0),  # (1, D, H, W)
            'forward_affine': forward_affine,  # (12,) - Changed from inverse_affine
            'inverse_affine': inverse_affine,
            'landmark_original': landmark_center_voxel.astype(float),
            'patch_shape': np.array([self.patch_size_vox, self.patch_size_vox, self.patch_size_vox])
        }
    
    def matrix_to_affine_params(self, matrix: np.ndarray) -> np.ndarray:
        """Flatten 4x4 affine matrix to 12 parameters (3x4)."""
        params = matrix[:3, :4].flatten()
        return params
    
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
