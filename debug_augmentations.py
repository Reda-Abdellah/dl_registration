import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import json
from src.dataset import RegistrationDatasetCTonly
from scipy.spatial.transform import Rotation as R
import argparse
import traceback


def matrix_to_euler_translation(matrix):
    """
    Convert 4x4 transformation matrix to Euler angles and translation.
    """
    if matrix is None:
        return {
            'rotation_x_deg': 0.0,
            'rotation_y_deg': 0.0, 
            'rotation_z_deg': 0.0,
            'translation_x_mm': 0.0,
            'translation_y_mm': 0.0,
            'translation_z_mm': 0.0
        }
    
    # Extract rotation matrix (3x3) and translation vector
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Convert rotation matrix to Euler angles
    try:
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
    except Exception as e:
        print(f"Warning: Could not extract Euler angles: {e}")
        euler_angles = np.array([0, 0, 0])
    
    return {
        'rotation_x_deg': float(euler_angles[0]),
        'rotation_y_deg': float(euler_angles[1]), 
        'rotation_z_deg': float(euler_angles[2]),
        'translation_x_mm': float(translation[0]),
        'translation_y_mm': float(translation[1]),
        'translation_z_mm': float(translation[2])
    }


def affine_params_to_matrix(params):
    """Convert 12-parameter affine to 4x4 matrix."""
    if params is None:
        return np.eye(4)
    
    if isinstance(params, torch.Tensor):
        params = params.numpy()
    
    params = params.reshape(3, 4)
    matrix = np.eye(4)
    matrix[:3, :4] = params
    return matrix


def safe_tensor_to_numpy(tensor):
    """Safely convert tensor to numpy, handling None and various tensor types."""
    if tensor is None:
        return None
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        
        # Handle different tensor dimensions
        if tensor.dim() == 4:  # (1, C, D, H, W) -> (C, D, H, W)
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 4 and tensor.shape[0] == 1:  # (1, D, H, W) -> (D, H, W)
            tensor = tensor.squeeze(0)
            
        return tensor.numpy()
    
    elif isinstance(tensor, np.ndarray):
        return tensor
    
    else:
        print(f"Warning: Unknown tensor type {type(tensor)}")
        return None


def test_single_sample(dataset, idx):
    """Test loading a single sample with detailed error reporting."""
    print(f"\n🔍 Testing sample {idx}...")
    
    try:
        # Get the raw data info first
        data_info = dataset.pairs[idx]
        ct_path = data_info['ct_path']
        print(f"  📁 CT Path: {ct_path}")
        print(f"  📁 Exists: {ct_path.exists()}")
        
        if ct_path.exists():
            print(f"  📏 File size: {ct_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Try to load the sample
        sample = dataset[idx]
        
        print("  ✅ Sample loaded successfully!")
        
        # Check each component
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"  📊 {key}: shape={value.shape}, dtype={value.dtype}")
                if value.numel() > 0:
                    print(f"      range=[{value.min().item():.3f}, {value.max().item():.3f}]")
                else:
                    print(f"      EMPTY TENSOR!")
            elif isinstance(value, np.ndarray):
                print(f"  📊 {key}: shape={value.shape}, dtype={value.dtype}")
                if value.size > 0:
                    print(f"      range=[{value.min():.3f}, {value.max():.3f}]")
                else:
                    print(f"      EMPTY ARRAY!")
            else:
                print(f"  📊 {key}: {type(value)} = {value}")
        
        return sample
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print(f"  📋 Traceback: {traceback.format_exc()}")
        return None


def visualize_sample_with_params(fixed, moving, transform_params, sample_idx, save_dir, warped=None):
    """
    Create visualization with robust error handling.
    """
    # Safely convert all inputs
    fixed_np = safe_tensor_to_numpy(fixed)
    moving_np = safe_tensor_to_numpy(moving)
    warped_np = safe_tensor_to_numpy(warped) if warped is not None else None
    
    # Check if we have valid data
    if fixed_np is None or moving_np is None:
        print(f"⚠️  Skipping visualization for sample {sample_idx}: invalid data")
        return None
    
    # Ensure 3D
    while fixed_np.ndim > 3:
        fixed_np = fixed_np.squeeze()
    while moving_np.ndim > 3:
        moving_np = moving_np.squeeze()
    
    if warped_np is not None:
        while warped_np.ndim > 3:
            warped_np = warped_np.squeeze()
    
    if fixed_np.ndim != 3 or moving_np.ndim != 3:
        print(f"⚠️  Invalid dimensions: fixed={fixed_np.shape}, moving={moving_np.shape}")
        return None
    
    # Get middle slices
    d, h, w = fixed_np.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.2)
    
    # Axial view (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(fixed_np[mid_d], cmap='gray')
    ax1.set_title('Fixed (Axial)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(moving_np[mid_d], cmap='gray')
    ax2.set_title('Moving (Axial)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    if warped_np is not None:
        ax3.imshow(warped_np[mid_d], cmap='gray')
        ax3.set_title('Warped (Axial)', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Warped\nImage', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Warped (Axial)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    target = warped_np if warped_np is not None else moving_np
    diff_axial = np.abs(fixed_np[mid_d] - target[mid_d])
    im4 = ax4.imshow(diff_axial, cmap='hot')
    ax4.set_title('|Fixed - Target|', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Sagittal view (middle row)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(fixed_np[:, :, mid_w], cmap='gray')
    ax5.set_title('Fixed (Sagittal)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(moving_np[:, :, mid_w], cmap='gray')
    ax6.set_title('Moving (Sagittal)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    if warped_np is not None:
        ax7.imshow(warped_np[:, :, mid_w], cmap='gray')
        ax7.set_title('Warped (Sagittal)', fontsize=12, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'No Warped\nImage', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Warped (Sagittal)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    diff_sag = np.abs(fixed_np[:, :, mid_w] - target[:, :, mid_w])
    im8 = ax8.imshow(diff_sag, cmap='hot')
    ax8.set_title('|Fixed - Target|', fontsize=12, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # Coronal view (bottom row)
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(fixed_np[:, mid_h, :], cmap='gray')
    ax9.set_title('Fixed (Coronal)', fontsize=12, fontweight='bold')
    ax9.axis('off')
    
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(moving_np[:, mid_h, :], cmap='gray')
    ax10.set_title('Moving (Coronal)', fontsize=12, fontweight='bold')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[2, 2])
    if warped_np is not None:
        ax11.imshow(warped_np[:, mid_h, :], cmap='gray')
        ax11.set_title('Warped (Coronal)', fontsize=12, fontweight='bold')
    else:
        ax11.text(0.5, 0.5, 'No Warped\nImage', ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Warped (Coronal)', fontsize=12, fontweight='bold')
    ax11.axis('off')
    
    ax12 = fig.add_subplot(gs[2, 3])
    diff_cor = np.abs(fixed_np[:, mid_h, :] - target[:, mid_h, :])
    im12 = ax12.imshow(diff_cor, cmap='hot')
    ax12.set_title('|Fixed - Target|', fontsize=12, fontweight='bold')
    ax12.axis('off')
    plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)
    
    # Add transformation parameters as text
    param_text = f"""Sample {sample_idx} - Transformation Parameters:
    
Rotation:    X: {transform_params['rotation_x_deg']:+7.2f}°    Y: {transform_params['rotation_y_deg']:+7.2f}°    Z: {transform_params['rotation_z_deg']:+7.2f}°
Translation: X: {transform_params['translation_x_mm']:+7.2f}mm   Y: {transform_params['translation_y_mm']:+7.2f}mm   Z: {transform_params['translation_z_mm']:+7.2f}mm

Image Stats: Fixed [{fixed_np.min():.3f}, {fixed_np.max():.3f}] | Moving [{moving_np.min():.3f}, {moving_np.max():.3f}]"""
    
    plt.figtext(0.02, 0.02, param_text, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Main title
    plt.suptitle(f'Dataset Debug - Sample {sample_idx}', fontsize=16, fontweight='bold')
    
    # Save figure
    save_path = Path(save_dir) / f'debug_sample_{sample_idx:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


def debug_augmentations(config_path='config.yaml', num_samples=10, output_dir='debug_output'):
    """
    Robust debug function with comprehensive error handling.
    """
    print("🔍 Starting augmentation debugging...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load dataset with error handling
    try:
        dataset = RegistrationDatasetCTonly(config_path)
        print(f"✅ Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return
    
    # Limit samples to dataset size
    num_samples = min(num_samples, len(dataset))
    
    # Debug info
    debug_info = {
        'config_path': config_path,
        'dataset_size': len(dataset),
        'samples_debugged': 0,
        'successful_samples': [],
        'failed_samples': []
    }
    
    print(f"📊 Processing {num_samples} samples...")
    
    # First, test a few samples to see what's wrong
    for i in range(min(3, num_samples)):
        test_single_sample(dataset, i)
    
    successful_count = 0
    
    for i in range(num_samples):
        try:
            print(f"\n📊 Processing sample {i+1}/{num_samples}...", end=' ')
            
            # Get sample from dataset
            sample = dataset[i]
            
            if sample is None:
                print("❌ Sample is None")
                debug_info['failed_samples'].append({
                    'sample_id': i,
                    'error': 'Sample is None'
                })
                continue
            
            # Extract data with error checking
            input_data = sample.get('input')
            fixed = sample.get('fixed')
            moving = sample.get('moving')
            
            if fixed is None or moving is None:
                print("❌ Missing fixed or moving data")
                debug_info['failed_samples'].append({
                    'sample_id': i,
                    'error': 'Missing fixed or moving data'
                })
                continue
            
            # Get transformation parameters
            affine_params = None
            if 'forward_affine' in sample:
                affine_params = sample['forward_affine']
            elif 'inverse_affine' in sample:
                # Convert inverse to forward for display
                inverse_matrix = affine_params_to_matrix(sample['inverse_affine'])
                forward_matrix = np.linalg.inv(inverse_matrix)
                affine_params = forward_matrix[:3, :4].flatten()
            
            if affine_params is None:
                print("⚠️  No affine parameters found, using identity")
                affine_params = np.array([1,0,0,0, 0,1,0,0, 0,0,1,0])
            
            # Convert to transformation matrix
            transform_matrix = affine_params_to_matrix(affine_params)
            
            # Extract rotation and translation
            transform_params = matrix_to_euler_translation(transform_matrix)
            
            # Create visualization
            save_path = visualize_sample_with_params(
                fixed,
                moving, 
                transform_params,
                i + 1,
                output_path
            )
            
            if save_path is not None:
                # Store sample info
                sample_info = {
                    'sample_id': i,
                    'image_path': str(save_path.relative_to(output_path)),
                    'transform_params': transform_params,
                    'affine_matrix': transform_matrix.tolist(),
                    'data_shapes': {
                        'input': list(input_data.shape) if input_data is not None else None,
                        'fixed': list(fixed.shape) if fixed is not None else None,
                        'moving': list(moving.shape) if moving is not None else None,
                    }
                }
                
                # Add landmark info if available
                for key in ['landmark_original', 'landmark_transformed', 'transform_center']:
                    if key in sample and sample[key] is not None:
                        sample_info[key] = sample[key].tolist() if hasattr(sample[key], 'tolist') else sample[key]
                
                debug_info['successful_samples'].append(sample_info)
                successful_count += 1
                print("✅")
            else:
                print("❌ Visualization failed")
                debug_info['failed_samples'].append({
                    'sample_id': i,
                    'error': 'Visualization failed'
                })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            debug_info['failed_samples'].append({
                'sample_id': i,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
    debug_info['samples_debugged'] = successful_count
    
    # Save debug info as JSON
    json_path = output_path / 'debug_info.json'
    with open(json_path, 'w') as f:
        json.dump(debug_info, f, indent=2)
    
    # Create summary if we have successful samples
    if debug_info['successful_samples']:
        create_summary_plot(debug_info, output_path)
    
    print(f"\n🎉 Debugging complete!")
    print(f"📁 Results saved to: {output_path}")
    print(f"✅ Successful samples: {successful_count}/{num_samples}")
    print(f"❌ Failed samples: {num_samples - successful_count}/{num_samples}")
    print(f"📊 Debug info: {json_path}")
    
    if debug_info['failed_samples']:
        print("\n⚠️  Failed samples summary:")
        for failed in debug_info['failed_samples']:
            print(f"   Sample {failed['sample_id']}: {failed['error']}")


def create_summary_plot(debug_info, output_dir):
    """Create summary plot of all transformations."""
    samples = debug_info['successful_samples']
    
    if not samples:
        return
    
    # Extract transformation parameters
    rotations_x = [s['transform_params']['rotation_x_deg'] for s in samples]
    rotations_y = [s['transform_params']['rotation_y_deg'] for s in samples]
    rotations_z = [s['transform_params']['rotation_z_deg'] for s in samples]
    
    translations_x = [s['transform_params']['translation_x_mm'] for s in samples]
    translations_y = [s['transform_params']['translation_y_mm'] for s in samples]
    translations_z = [s['transform_params']['translation_z_mm'] for s in samples]
    
    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Rotation histograms
    axes[0, 0].hist(rotations_x, bins=10, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_title('Rotation X Distribution')
    axes[0, 0].set_xlabel('Degrees')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(rotations_y, bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Rotation Y Distribution')
    axes[0, 1].set_xlabel('Degrees')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(rotations_z, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].set_title('Rotation Z Distribution')
    axes[0, 2].set_xlabel('Degrees')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Translation histograms
    axes[1, 0].hist(translations_x, bins=10, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('Translation X Distribution')
    axes[1, 0].set_xlabel('mm')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(translations_y, bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Translation Y Distribution')
    axes[1, 1].set_xlabel('mm')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(translations_z, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 2].set_title('Translation Z Distribution')
    axes[1, 2].set_xlabel('mm')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Transformation Parameters Summary ({len(samples)} samples)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(output_dir) / 'transformation_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Debug dataset augmentations with robust error handling')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to debug')
    parser.add_argument('--output', type=str, default='debug_output', help='Output directory')
    
    args = parser.parse_args()
    
    debug_augmentations(
        config_path=args.config,
        num_samples=args.samples,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

#python debug_augmentations.py --config config.yaml --samples 10 --output debug_output
