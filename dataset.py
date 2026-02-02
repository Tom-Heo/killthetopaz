import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from heo import Heo

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, phase='train', crop_size=128):
        """
        Args:
            root_dir (str): Root directory of DIV2K dataset.
                            Expected structure:
                            root_dir/DIV2K_train_HR/
            phase (str): 'train' or 'valid'.
            crop_size (int): Size of the random crop.
        """
        self.root_dir = root_dir
        self.phase = phase
        self.crop_size = crop_size
        
        if phase == 'train':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_train_HR')
        else:
            self.hr_dir = os.path.join(root_dir, 'DIV2K_valid_HR')
            
        self.image_paths = sorted(glob.glob(os.path.join(self.hr_dir, '*.png')))
        self.srgb_to_oklab = Heo.sRGBtoOklab()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load HR image
        img_path = self.image_paths[idx]
        hr_img = Image.open(img_path).convert('RGB')
        
        # 2. Random Crop
        if self.phase == 'train':
            # Ensure image is large enough
            w, h = hr_img.size
            if w < self.crop_size or h < self.crop_size:
                # Resize if too small (rare for DIV2K but safe)
                hr_img = TF.resize(hr_img, self.crop_size)
            
            i, j, h, w = transforms.RandomCrop.get_params(
                hr_img, output_size=(self.crop_size, self.crop_size)
            )
            hr_img = TF.crop(hr_img, i, j, h, w)
        else:
            # For validation, maybe center crop or just return full/resized
            # Let's do center crop for consistency
            hr_img = TF.center_crop(hr_img, self.crop_size)

        # 3. Generate LR (Downsample x4 -> Upsample x4)
        # Using Bicubic as per plan
        w, h = hr_img.size
        lr_img = hr_img.resize((w // 4, h // 4), Image.BICUBIC)
        lr_img = lr_img.resize((w, h), Image.BICUBIC) # Pre-upsample

        # 4. To Tensor [0, 1]
        hr_tensor = TF.to_tensor(hr_img)
        lr_tensor = TF.to_tensor(lr_img)
        
        # 5. Convert to Oklab
        # sRGBtoOklab expects (B, 3, H, W), so unsqueeze and squeeze
        # But we can't depend on batch dim here easily if we want to run this in Dataset.
        # Actually Heo.sRGBtoOklab is an nn.Module, meant for batches.
        # We can apply it here or in the training loop.
        # Applying here is cleaner for the loop but slower if not batched? 
        # No, dataset is per item. We unsqueeze(0).
        
        with torch.no_grad():
            hr_oklab = self.srgb_to_oklab(hr_tensor.unsqueeze(0)).squeeze(0)
            lr_oklab = self.srgb_to_oklab(lr_tensor.unsqueeze(0)).squeeze(0)
            
        return {'hr': hr_oklab, 'lr': lr_oklab}
