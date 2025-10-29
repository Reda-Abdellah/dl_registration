import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os
from pathlib import Path
from dataset import RegistrationDataset, RegistrationDatasetCTonly
from model import STN
from loss import RegistrationLoss
import matplotlib.pyplot as plt  # For 3D viz, use nibabel + matplotlib

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['model']['device'])
    
    # Dataset and loader
    dataset = RegistrationDatasetCTonly('config.yaml')
    loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True)
    
    # Model
    model = STN(in_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    loss_fn = RegistrationLoss(config)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(config['model']['epochs']):
        model.train()
        epoch_loss = 0
        for batch in tqdm(loader):
            input = batch['input'].to(device)
            fixed = input[:, 0]
            true_affine = batch['inverse_affine'].to(device)
            
            optimizer.zero_grad()
            warped, pred_affine = model(input)
            
            loss = loss_fn(pred_affine, true_affine, fixed, warped[:, :, :, :, 0])  # Use one channel for MI demo
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config['model']['epochs']}, Loss: {epoch_loss/len(loader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")
        
        # Optional: Visualize slice
        if epoch % 10 == 0:
            visualize_slice(fixed[0], fixed[1], warped[0], epoch)

def visualize_slice(fixed, moving, warped, epoch):
    # Plot middle axial slice
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(fixed[0, 32].cpu().numpy(), cmap='gray')  # Fixed CT/T2 slice
    axs[0].set_title('Fixed')
    axs[1].imshow(moving[0, 32].cpu().numpy(), cmap='gray')
    axs[1].set_title('Moving (Transformed)')
    axs[2].imshow(warped[0, 32].cpu().numpy(), cmap='gray')
    axs[2].set_title('Warped Back')
    plt.savefig(f'vis_epoch_{epoch}.png')
    plt.close()

if __name__ == "__main__":
    main()
