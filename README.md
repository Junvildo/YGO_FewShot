# YGO_FewShot: Few-Shot Yu-Gi-Oh Card Recognition

A metric learning pipeline for few-shot Yu-Gi-Oh card recognition using MobileOne embeddings, YOLO detection, and FAISS-based retrieval.

## Project Overview

This repository implements an end-to-end card recognition system that:
1. **Detects** cards in real-world images using YOLO
2. **Preprocesses** cards through perspective correction and artwork extraction
3. **Extracts** discriminative embeddings using MobileOne backbone with metric learning
4. **Retrieves** similar cards from a database using FAISS similarity search

The approach combines efficient MobileOne embeddings with metric learning losses to enable accurate card recognition with minimal training data.

## Architecture

### Core Components

- **MobileOne** (`mobileone.py`): Lightweight efficient CNN backbone for feature extraction
- **EmbeddedFeatureWrapper** (`models.py`): Wrapper adding embedding layer and GeM pooling on top of MobileOne
- **Metric Learning Loss** (`losses.py`): NormSoftmaxLoss with temperature scaling for training embeddings
- **YOLO Detection** (`app.py`): Pre-trained YOLO for card detection in field images
- **FAISS Retrieval** (`retrieval.py`): Fast similarity search for card matching
- **Data Preprocessing** (`data_preprocess.py`): Perspective correction and artwork extraction

### Training Pipeline

1. **Pretraining** (Epoch 0): Initialize and pretrain the new embedding layer with frozen backbone
2. **Finetuning** (Epoch 1+): Finetune entire model with adjusted learning rate
3. **Evaluation**: Extract features and evaluate via FAISS-based retrieval metrics

Key features:
- ClassBalancedBatchSampler for handling imbalanced classes
- Cosine annealing learning rate scheduler
- Lion optimizer support
- Data augmentation: RandomResizedCrop, ColorJitter, GaussianBlur, AffineTransform, Perspective, Erasing

### Inference Pipeline (`app.py`)

1. Load field image with multiple cards
2. Detect cards using YOLO
3. Apply perspective correction to each detected card
4. Extract artwork region
5. Extract embeddings from preprocessed artwork
6. Retrieve top-k matches from database using FAISS

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

The code uses the following environment variable (automatically set in entry points):
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Training

```bash
python train.py --model_variant s2 --batch_size 32 --dim 512 --output ./finetuned_models/
```

Key arguments:
- `--model_variant`: MobileOne variant (s0, s1, s2, s3, s4)
- `--batch_size`: Batch size
- `--dim`: Output embedding dimension
- `--output`: Output directory for checkpoints
- `--dataset_root`: Path to dataset (ImageFolder structure)
- `--epochs`: Number of epochs

### Evaluation

```bash
python eval.py --snap ./finetuned_models/s2_56_epoch_45.pth --dataset_root ./main_dataset
```

Extracts embeddings and evaluates retrieval metrics using FAISS.

### Inference/Demo

```bash
python app.py
```

Requires:
- YOLO model at `finetuned_models/yolo_ygo.pt`
- Input image at `real_life_images/ygo_field2.jpg`
- Card embedding database

## Dataset & Model Assumptions

- **Image Size**: Default 56×56 for embeddings
- **Dataset Format**: ImageFolder structure (class folders containing images)
- **Checkpoints**: Saved under `finetuned_models/`

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training entry point with pretraining and finetuning stages |
| `eval.py` | Extract embeddings and evaluate retrieval performance |
| `app.py` | End-to-end inference demo with detection and retrieval |
| `mobileone.py` | MobileOne backbone architecture with reparameterization |
| `models.py` | EmbeddedFeatureWrapper and GeM pooling layers |
| `losses.py` | NormSoftmaxLoss for metric learning |
| `arcface.py` | ArcFace loss implementation |
| `sampler.py` | ClassBalancedBatchSampler for balanced training |
| `data.py` | Dataset loaders and transforms |
| `data_preprocess.py` | Perspective correction and artwork extraction utilities |
| `retrieval.py` | FAISS-based embedding retrieval and evaluation |
| `extract_features.py` | Feature extraction utilities |
| `util.py` | Logging and plotting utilities |

## Demo

[View Demo](https://github.com/user-attachments/assets/6ff0638a-b99b-4433-a0ca-83ecbc51116c)

