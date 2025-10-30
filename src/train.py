import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os
from pathlib import Path
from datetime import datetime
import json
import shutil
import argparse
from src.dataset import RegistrationDatasetCTonly
from src.model import STN
from src.loss import RegistrationLoss
import matplotlib.pyplot as plt


class ExperimentManager:
    """Manages experiment tracking, checkpointing, and organization."""
    
    def __init__(self, config, experiment_name=None, resume_from=None):
        """
        Initialize experiment manager.
        
        Args:
            config: Configuration dictionary
            experiment_name: Optional name for this experiment run
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.resume_from = resume_from
        
        # Create experiment directory structure
        if resume_from:
            # Resume from existing experiment
            self.run_dir = Path(resume_from).parent.parent
            self.experiment_name = self.run_dir.name
            print(f"Resuming experiment: {self.experiment_name}")
        else:
            # Create new experiment
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = experiment_name or f"exp_{timestamp}"
            
            base_dir = Path(config.get('experiment', {}).get('base_dir', 'experiments'))
            self.run_dir = base_dir / self.experiment_name
            self.run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created new experiment: {self.experiment_name}")
        
        # Create subdirectories
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.visualization_dir = self.run_dir / 'visualizations'
        self.logs_dir = self.run_dir / 'logs'
        
        for d in [self.checkpoint_dir, self.visualization_dir, self.logs_dir]:
            d.mkdir(exist_ok=True)
        
        # Manifest file
        self.manifest_path = self.run_dir / 'manifest.json'
        
        if not resume_from:
            # Initialize new manifest
            self.manifest = self._create_manifest()
            self._save_manifest()
            
            # Save configuration
            self._save_config()
        else:
            # Load existing manifest
            self.manifest = self._load_manifest()
    
    def _create_manifest(self):
        """Create experiment manifest with metadata."""
        return {
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat(),
            'status': 'running',
            'config': self.config,
            'runs': [],
            'best_metrics': {
                'epoch': 0,
                'loss': float('inf'),
                'checkpoint': None
            },
            'metadata': {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device': str(self.config['model']['device'])
            }
        }
    
    def _load_manifest(self):
        """Load existing manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return self._create_manifest()
    
    def _save_manifest(self):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _save_config(self):
        """Save configuration to experiment directory."""
        config_path = self.run_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def update_manifest(self, updates):
        """Update manifest with new information."""
        self.manifest.update(updates)
        self.manifest['updated_at'] = datetime.now().isoformat()
        self._save_manifest()
    
    def log_epoch(self, epoch, metrics):
        """Log epoch metrics to manifest."""
        run_info = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self.manifest['runs'].append(run_info)
        
        # Update best metrics
        if metrics['total_loss'] < self.manifest['best_metrics']['loss']:
            self.manifest['best_metrics'].update({
                'epoch': epoch,
                'loss': metrics['total_loss'],
                'checkpoint': f"epoch_{epoch}.pth"
            })
        
        self._save_manifest()
    
    def get_checkpoint_path(self, epoch):
        """Get checkpoint path for specific epoch."""
        return self.checkpoint_dir / f"epoch_{epoch}.pth"
    
    def get_best_checkpoint_path(self):
        """Get path to best checkpoint."""
        best_ckpt = self.manifest['best_metrics']['checkpoint']
        if best_ckpt:
            return self.checkpoint_dir / best_ckpt
        return None
    
    def finalize(self, status='completed'):
        """Finalize experiment."""
        self.manifest['status'] = status
        self.manifest['completed_at'] = datetime.now().isoformat()
        self._save_manifest()
        print(f"\nExperiment finalized with status: {status}")
        print(f"Results saved in: {self.run_dir}")


def save_checkpoint(model, optimizer, epoch, loss, history, exp_manager, is_best=False):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'history': history,
        'config': exp_manager.config,
        'experiment_name': exp_manager.experiment_name
    }
    
    # Save regular checkpoint
    checkpoint_path = exp_manager.get_checkpoint_path(epoch)
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = exp_manager.checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"  ✓ Best model saved: {best_path}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint and restore training state."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  ✓ Model state loaded")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  ✓ Optimizer state loaded")
    
    start_epoch = checkpoint.get('epoch', 0)
    history = checkpoint.get('history', {'train_loss': [], 'affine_loss': [], 'similarity_loss': []})
    config = checkpoint.get('config', {})
    
    print(f"  ✓ Resuming from epoch {start_epoch}")
    
    return start_epoch, history, config


def main(config_path='config.yaml', experiment_name=None, resume_from=None, device=None):
    """
    Main training function - can be called programmatically or from command line.
    
    Args:
        config_path: Path to configuration YAML file
        experiment_name: Optional experiment name (default: timestamp-based)
        resume_from: Optional path to checkpoint to resume from
        device: Optional device override ('cuda' or 'cpu')
    
    Returns:
        Dictionary with training results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if device:
        config['model']['device'] = device
    
    device = torch.device(config['model']['device'])
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(config, experiment_name=experiment_name, resume_from=resume_from)
    
    # Dataset and loader
    dataset = RegistrationDatasetCTonly(config_path)
    loader = DataLoader(
        dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model and optimizer
    model = STN(in_channels=2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['model']['learning_rate'],
        weight_decay=config['model'].get('weight_decay', 1e-5)
    )
    loss_fn = RegistrationLoss(config).to(device)
    
    # Load checkpoint if resuming
    start_epoch = 0
    history = {'train_loss': [], 'affine_loss': [], 'similarity_loss': []}
    
    if resume_from:
        start_epoch, history, _ = load_checkpoint(resume_from, model, optimizer)
    
    # Training loop
    best_loss = float('inf')
    if history['train_loss']:
        best_loss = min(history['train_loss'])
    
    total_epochs = config['model']['epochs']
    
    try:
        for epoch in range(start_epoch, total_epochs):
            model.train()
            epoch_loss = 0
            epoch_affine_loss = 0
            epoch_similarity_loss = 0
            
            progress_bar = tqdm(
                loader, 
                desc=f"Epoch {epoch+1}/{total_epochs}",
                leave=True
            )
            
            # In the training loop, replace the forward pass section:
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                input_data = batch['input'].to(device)
                fixed = batch['fixed'].to(device)
                moving = batch['moving'].to(device)
                # Use forward transformation instead of inverse
                true_affine = batch['forward_affine'].to(device)  # Changed from inverse_affine
                
                # Forward pass
                optimizer.zero_grad()
                warped, pred_affine = model(input_data)
                
                # Compute loss using fixed and warped images
                total_loss, loss_dict = loss_fn(pred_affine, true_affine, fixed, warped)
                
                # Extract single channel from warped
                warped_single = warped[:, 1:2]

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate losses
                epoch_loss += loss_dict['total_loss']
                epoch_affine_loss += loss_dict['affine_loss']
                epoch_similarity_loss += loss_dict['similarity_loss']
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'affine': f"{loss_dict['affine_loss']:.4f}",
                    'sim': f"{loss_dict['similarity_loss']:.4f}"
                })
                
                # Free memory periodically
                if batch_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate epoch averages
            avg_loss = epoch_loss / len(loader)
            avg_affine = epoch_affine_loss / len(loader)
            avg_similarity = epoch_similarity_loss / len(loader)
            
            # Store history
            history['train_loss'].append(avg_loss)
            history['affine_loss'].append(avg_affine)
            history['similarity_loss'].append(avg_similarity)
            
            # Log to manifest
            metrics = {
                'total_loss': avg_loss,
                'affine_loss': avg_affine,
                'similarity_loss': avg_similarity
            }
            exp_manager.log_epoch(epoch + 1, metrics)
            
            print(f"\nEpoch {epoch+1}/{total_epochs} Summary:")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Affine Loss: {avg_affine:.4f}")
            print(f"  Similarity Loss: {avg_similarity:.4f}")
            
            # Check if best model
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                print(f"  ✓ New best loss: {best_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch + 1, avg_loss, history, 
                exp_manager, is_best=is_best
            )
            print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Visualize every N epochs
            viz_freq = config.get('visualization', {}).get('frequency', 5)
            if epoch % viz_freq == 0 or is_best:
                with torch.no_grad():
                    model.eval()
                    visualize_registration(
                        fixed[0], 
                        moving[0], 
                        warped_single[0], 
                        epoch + 1,
                        save_dir=exp_manager.visualization_dir
                    )
                    model.train()
            
            # Plot training history
            if (epoch + 1) % 5 == 0:
                plot_training_history(
                    history, 
                    save_path=exp_manager.visualization_dir / 'training_history.png'
                )
        
        # Training completed successfully
        exp_manager.finalize(status='completed')
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        exp_manager.finalize(status='interrupted')
        
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        exp_manager.finalize(status='failed')
        raise
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Experiment: {exp_manager.experiment_name}")
    print(f"Location: {exp_manager.run_dir}")
    print(f"Best loss: {best_loss:.4f} at epoch {exp_manager.manifest['best_metrics']['epoch']}")
    print(f"Best checkpoint: {exp_manager.get_best_checkpoint_path()}")
    print("="*80)
    
    # Return results for programmatic use
    return {
        'experiment_name': exp_manager.experiment_name,
        'run_dir': str(exp_manager.run_dir),
        'best_loss': best_loss,
        'best_epoch': exp_manager.manifest['best_metrics']['epoch'],
        'best_checkpoint': str(exp_manager.get_best_checkpoint_path()),
        'history': history,
        'manifest': exp_manager.manifest
    }


def visualize_registration(fixed, moving, warped, epoch, save_dir='visualizations'):
    """Visualize registration results for a single sample."""
    fixed_np = fixed.cpu().squeeze(0).numpy()
    moving_np = moving.cpu().squeeze(0).numpy()
    warped_np = warped.cpu().squeeze(0).numpy()
    
    d, h, w = fixed_np.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Axial slices
    axes[0, 0].imshow(fixed_np[mid_d], cmap='gray')
    axes[0, 0].set_title('Fixed (Axial)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(moving_np[mid_d], cmap='gray')
    axes[0, 1].set_title('Moving (Axial)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(warped_np[mid_d], cmap='gray')
    axes[0, 2].set_title('Warped (Axial)')
    axes[0, 2].axis('off')
    
    diff = np.abs(fixed_np[mid_d] - warped_np[mid_d])
    axes[0, 3].imshow(diff, cmap='hot')
    axes[0, 3].set_title('|Fixed - Warped|')
    axes[0, 3].axis('off')
    
    # Sagittal slices
    axes[1, 0].imshow(fixed_np[:, :, mid_w], cmap='gray')
    axes[1, 0].set_title('Fixed (Sagittal)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(moving_np[:, :, mid_w], cmap='gray')
    axes[1, 1].set_title('Moving (Sagittal)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(warped_np[:, :, mid_w], cmap='gray')
    axes[1, 2].set_title('Warped (Sagittal)')
    axes[1, 2].axis('off')
    
    diff_sag = np.abs(fixed_np[:, :, mid_w] - warped_np[:, :, mid_w])
    axes[1, 3].imshow(diff_sag, cmap='hot')
    axes[1, 3].set_title('|Fixed - Warped|')
    axes[1, 3].axis('off')
    
    # Coronal slices
    axes[2, 0].imshow(fixed_np[:, mid_h, :], cmap='gray')
    axes[2, 0].set_title('Fixed (Coronal)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(moving_np[:, mid_h, :], cmap='gray')
    axes[2, 1].set_title('Moving (Coronal)')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(warped_np[:, mid_h, :], cmap='gray')
    axes[2, 2].set_title('Warped (Coronal)')
    axes[2, 2].axis('off')
    
    diff_cor = np.abs(fixed_np[:, mid_h, :] - warped_np[:, mid_h, :])
    axes[2, 3].imshow(diff_cor, cmap='hot')
    axes[2, 3].set_title('|Fixed - Warped|')
    axes[2, 3].axis('off')
    
    plt.suptitle(f'Registration Results - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    save_path = Path(save_dir) / f'registration_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """Plot training loss history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Component losses
    axes[1].plot(epochs, history['affine_loss'], 'r-', linewidth=2, label='Affine Loss')
    axes[1].plot(epochs, history['similarity_loss'], 'g-', linewidth=2, label='Similarity Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main_cli():
    """Command-line interface wrapper for main()."""
    parser = argparse.ArgumentParser(description='Train registration model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Call main with parsed arguments
    main(
        config_path=args.config,
        experiment_name=args.name,
        resume_from=args.resume,
        device=args.device
    )


if __name__ == "__main__":
    main_cli()
