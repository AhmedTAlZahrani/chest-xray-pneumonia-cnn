import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from .dataset import ChestXRayDataset
from .model import load_model


@torch.no_grad()
def evaluate(model, loader, device):
    """Run inference on a dataset and collect predictions."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test directory")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="output/metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device=device)

    test_ds = ChestXRayDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Test set: {len(test_ds)} images")

    labels, preds, probs = evaluate(model, test_loader, device)

    target_names = ChestXRayDataset.CLASS_NAMES
    report = classification_report(labels, preds, target_names=target_names, output_dict=True)
    cm = confusion_matrix(labels, preds)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names))
    print("Confusion Matrix:")
    print(cm)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": report["accuracy"],
        "per_class": {name: report[name] for name in target_names},
        "confusion_matrix": cm.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
