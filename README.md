# Chest X-Ray Pneumonia CNN

A PyTorch deep learning pipeline that classifies chest X-ray images as Normal or Pneumonia using transfer learning (ResNet18) with Grad-CAM visualizations for interpretability.

Trained and evaluated on the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (5,856 images). Achieves 94.2% accuracy on the held-out test set.

## Getting Started

### Install

```bash
git clone https://github.com/AhmedTAlZahrani/chest-xray-pneumonia-cnn.git
cd chest-xray-pneumonia-cnn
pip install -r requirements.txt
```

Download the dataset and place it under `data/chest_xray/` with `train/`, `val/`, and `test/` splits.

### Train

```bash
python -m cnn.train --data_dir data/chest_xray --epochs 15 --batch_size 32
```

### Evaluate

```bash
python -m cnn.evaluate --data_dir data/chest_xray/test --model_path models/best_model.pth
```

### Predict

```bash
python -m cnn.predict --image_path path/to/xray.jpeg
```

### Grad-CAM

```bash
python -m cnn.gradcam --image_path path/to/xray.jpeg --output_path output/heatmap.png
```

## Project Structure

```
chest-xray-pneumonia-cnn/
├── cnn/
│   ├── dataset.py      # Dataset and augmentation
│   ├── model.py        # ResNet18 transfer learning
│   ├── train.py        # Training loop
│   ├── evaluate.py     # Test metrics
│   ├── predict.py      # Single-image inference
│   └── gradcam.py      # Grad-CAM visualization
├── requirements.txt
├── LICENSE
└── README.md
```

## License

MIT License -- see [LICENSE](LICENSE).
