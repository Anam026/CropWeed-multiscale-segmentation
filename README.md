# Crop-Weed Segmentation with MSFCA + UNet++ Pipeline

Multiscale semantic segmentation project for crop/weed discrimination using the WeedsGalore dataset.

## Architecture Overview

```
Input RGB Image
    ↓
Data Preprocessing (Resize, Normalize, Multi-Scale Augmentation)
    ↓
CNN Encoder (ResNet50 / VGG16)
    ↓
MSFCA Module (Multi-Scale Feature Channel Attention)
    → 1×5 / 5×1, 1×11 / 11×1, 1×17 / 17×1 convolutions
    ↓
Global Transformer (Global Context Modeling)
    ↓
Fusion Module (MSFCA + Transformer features)
    ↓
UNet++ Decoder (Nested Skip Connections) + Edge Detection
    ↓
Final Segmentation Output
    ↓
Loss: L_total = L_seg + 0.3 * L_edge
       (Dice + Focal + Edge Loss)
```

## Project Structure

```
crop_weed_segmentation/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── data/
│   ├── raw/                     # Downloaded WeedsGalore dataset
│   └── processed/               # Train/val/test splits
│       ├── train/{images,masks}
│       ├── val/{images,masks}
│       └── test/{images,masks}
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # WeedsGalore dataset class
│   │   └── transforms.py        # Augmentations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py           # ResNet/VGG CNN encoder
│   │   ├── msfca.py             # Multi-Scale Feature Channel Attention
│   │   ├── transformer.py       # Global Transformer module
│   │   ├── unetpp_decoder.py    # UNet++ with nested skip connections
│   │   ├── edge_detection.py    # Edge detection head
│   │   └── segmentation_model.py# Full pipeline
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py            # Dice + Focal + Edge losses
│   │   └── trainer.py           # Training loop
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # IoU, Dice, Precision, Recall
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py     # Result plotting
│       └── helpers.py           # Utility functions
├── scripts/
│   ├── prepare_dataset.py       # Download & preprocess WeedsGalore
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── infer.py                 # Single-image inference
├── notebooks/
│   └── exploration.ipynb        # Dataset EDA notebook
├── outputs/
│   ├── checkpoints/             # Saved model weights
│   ├── logs/                    # TensorBoard / CSV logs
│   └── visualizations/          # Prediction overlays
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and prepare dataset
```bash
python scripts/prepare_dataset.py --dataset_url https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip --output_dir data/raw
```

### 3. Train
```bash
python scripts/train.py --config configs/config.yaml
```

### 4. Evaluate
```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

### 5. Inference
```bash
python scripts/infer.py --image path/to/image.png --checkpoint outputs/checkpoints/best_model.pth
```

## Dataset: WeedsGalore

- **Source**: https://doidata.gfz.de/weedsgalore_e_celikkan_2024/
- **Structure**: Date-organized folders (2023-05-25, 2023-05-30, 2023-06-06, 2023-06-15)
- **Per folder**: `images/` (B, G, R, NIR, RE bands), `semantics/` (semantic masks), `instances/`
- **Bands used**: R, G, B (RGB composite) — extendable to NIR/RE for multispectral
- **Classes**: background, crop, weed

## Loss Function

```
L_total = L_seg + 0.3 * L_edge
L_seg   = 0.5 * L_dice + 0.5 * L_focal
```

## Key Hyperparameters (see configs/config.yaml)

| Parameter      | Value    |
|----------------|----------|
| Encoder        | ResNet50 |
| Image size     | 512×512  |
| Batch size     | 8        |
| Learning rate  | 1e-4     |
| Epochs         | 100      |
| Edge weight    | 0.3      |
| Num classes    | 3        |
