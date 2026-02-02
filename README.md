# BakeNet: Perceptually Optimized Image Super-Resolution

BakeNet is an AI model designed to restore lost color information and upscale images, focusing on perceptual accuracy using the Oklab color space.

## 🚀 Quick Start (Cloud Environment)

### 1. Setup Environment
Clone the repository and run the setup script to install dependencies.

```bash
git clone https://github.com/Tom-Heo/killthetopaz.git
cd killthetopaz
bash setup.sh
```

### 2. Prepare Dataset (DIV2K)
Download and unzip the DIV2K dataset automatically.

```bash
bash download_data.sh
```

---

## 🛠 Usage

### Training

Start training from scratch:
```bash
python train.py --data_root ./data
```

**Common Options:**
*   `--batch_size`: Batch size (default: 16). Reduce if OOM occurs.
*   `--epochs`: Number of epochs (default: 100).
*   `--save_interval`: Checkpoint save interval (default: 10 epochs).

**Resume Training:**
To continue training from a saved checkpoint:
```bash
python train.py --data_root ./data --resume checkpoints/bake_last.pth
```

### Inference (Upscaling)

Upscale an image using a trained model.

**Single Image:**
```bash
python inference.py --input "test.png" --checkpoint "checkpoints/bake_best.pth" --pre_upsample
```

**Directory of Images:**
```bash
python inference.py --input "images_folder/" --checkpoint "checkpoints/bake_best.pth" --pre_upsample --output_dir "results/"
```

*   `--pre_upsample`: Use this if your input is a small low-resolution image (e.g., thumbnail). It performs a bicubic x4 upscale before feeding it to the model.
*   If your input is already upscaled (blurry but large), omit `--pre_upsample`.

---

## 📂 Project Structure

*   `heo.py`: Core model definitions (BakeNet, BakeDDPM, HeLU, BakedColor).
*   `train.py`: Training script with EMA and AdamW.
*   `inference.py`: Inference script for upscaling.
*   `dataset.py`: DIV2K dataset loader with Oklab conversion.
*   `setup.sh`: Auto-setup script for Linux environments.
*   `download_data.sh`: DIV2K downloader.
