import os
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from heo import Heo

@torch.no_grad()
def inference(args):
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Model Initialization
    network = Heo.BakeNet(in_channels=3, dim=96, num_blocks=20)
    model = Heo.BakeDDPM(network, timesteps=args.timesteps)
    
    # 3. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load EMA model if available (preferred for inference), else regular model
    if 'ema_state_dict' in checkpoint:
        print("Loading EMA state dict...")
        model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        print("Loading standard model state dict...")
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model = model.to(device)
    model.eval()

    # 4. Data Preparation
    if os.path.isdir(args.input):
        input_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        input_paths = [args.input]
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Converters
    srgb_to_oklab = Heo.sRGBtoOklab().to(device)
    oklab_to_srgb = Heo.OklabtosRGB().to(device)

    # 5. Inference Loop
    for img_path in tqdm(input_paths, desc="Processing Images"):
        # Load Image
        img_name = os.path.basename(img_path)
        lr_img = Image.open(img_path).convert('RGB')
        
        # Pre-upsample (Bicubic x4) if needed (assuming input is LR)
        if args.pre_upsample:
            w, h = lr_img.size
            lr_img = lr_img.resize((w * 4, h * 4), Image.BICUBIC)
            
        # To Tensor & Oklab
        lr_tensor = TF.to_tensor(lr_img).unsqueeze(0).to(device) # (1, 3, H, W)
        lr_oklab = srgb_to_oklab(lr_tensor)
        
        # Diffusion Sampling
        # Output is (1, 3, H, W) in Oklab space
        sr_oklab = model.sample(lr_oklab)
        
        # Oklab to sRGB
        sr_srgb = oklab_to_srgb(sr_oklab)
        
        # Save Result
        sr_img = TF.to_pil_image(sr_srgb.squeeze(0).cpu())
        save_path = os.path.join(args.output_dir, f"bake_{img_name}")
        sr_img.save(save_path)
        
    print(f"Inference complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BakeNet Inference")
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps (must match training)')
    parser.add_argument('--pre_upsample', action='store_true', help='Upsample input image x4 using Bicubic before inference')

    args = parser.parse_args()
    inference(args)
