import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from heo import Heo
from dataset import DIV2KDataset

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

def train(args):
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Dataset & DataLoader
    train_dataset = DIV2KDataset(root_dir=args.data_root, phase='train', crop_size=args.crop_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    valid_dataset = DIV2KDataset(root_dir=args.data_root, phase='valid', crop_size=args.crop_size)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Valid Dataset Size: {len(valid_dataset)}")

    # 3. Model Initialization
    network = Heo.BakeNet(in_channels=3, dim=96, num_blocks=20)
    model = Heo.BakeDDPM(network, timesteps=args.timesteps)
    model = model.to(device)
    
    # Initialize EMA
    ema = EMA(model, decay=args.ema_decay)
    ema.ema_model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Calculate Gamma if not provided or to target a final LR
    if args.target_final_lr:
        total_steps = args.epochs * len(train_loader)
        calculated_gamma = (args.target_final_lr / args.lr) ** (1 / total_steps)
        print(f"Calculated Gamma for target final LR {args.target_final_lr}: {calculated_gamma:.6f}")
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=calculated_gamma)
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # 5. Training Loop
    start_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])
            
            if 'loss' in checkpoint:
                # If resuming from best model, we might want to keep track of that best loss
                # But usually we resume from 'last', so we might not know the absolute best loss unless stored separately.
                # However, if we just want to continue training, this is fine. 
                # If we resume from 'best', current loss is best.
                # Let's just trust the loop to find new bests.
                pass
            
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        train_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            x_hr = batch['hr'].to(device)
            x_lr = batch['lr'].to(device)

            optimizer.zero_grad()
            
            # Forward (Calculates Loss)
            loss = model(x_hr, x_lr)
            
            loss.backward()
            
            # No Gradient Clipping
            
            optimizer.step()
            
            # Step Scheduler per iteration
            scheduler.step()
            
            # Update EMA
            ema.update(model)

            train_loss += loss.item()
            
            # Log every 100 steps
            global_step = epoch * len(train_loader) + batch_idx
            if global_step % 100 == 0:
                print(f"Step {global_step}: Loss = {loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.8f}")
                
            # Save Checkpoint every 1000 steps
            if global_step > 0 and global_step % 1000 == 0:
                step_save_path = os.path.join(args.save_dir, f"bake_step_{global_step}.pth")
                torch.save({
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, step_save_path)
                print(f"Checkpoint saved to {step_save_path}")
                
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop (using EMA model)
        ema.ema_model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                x_hr = batch['hr'].to(device)
                x_lr = batch['lr'].to(device)
                loss = ema.ema_model(x_hr, x_lr)
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.6f}, Valid Loss (EMA): {avg_valid_loss:.6f}")

        # 6. Save Checkpoints
        
        # Save Last
        last_path = os.path.join(args.save_dir, "bake_last.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
        }, last_path)
        
        # Save Best (Based on Valid Loss)
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_path = os.path.join(args.save_dir, "bake_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, best_path)
            print(f"New best model saved to {best_path} (Valid Loss: {best_loss:.6f})")

        # Save Interval (Epoch based - kept for backward compatibility or if preferred)
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"bake_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
            }, save_path)
            print(f"Epoch Checkpoint saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BakeNet on DIV2K")
    parser.add_argument('--data_root', type=str, default='./data', help='Path to DIV2K root directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--crop_size', type=int, default=128, help='Random crop size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--gamma', type=float, default=0.9999, help='Exponential LR decay gamma per step')
    parser.add_argument('--target_final_lr', type=float, default=1e-6, help='Target final learning rate (overrides gamma)')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval in epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()
    train(args)
