import argparse

import torch
from PIL import Image

from .dataset import get_transforms, ChestXRayDataset
from .model import load_model


def predict_image(model, image_path, device="cpu"):
    """Run inference on a single image.

    Args:
        model: Trained model in eval mode.
        image_path: Path to the input image.
        device: Device to run inference on.

    Returns:
        Tuple of (predicted_class, confidence, probabilities).
    """
    transform = get_transforms(split="test")
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        confidence, predicted = probs.max(0)

    class_name = ChestXRayDataset.CLASS_NAMES[predicted.item()]
    return class_name, confidence.item(), probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Predict on a single chest X-ray image")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device=device)

    class_name, confidence, probs = predict_image(model, args.image_path, device)

    print(f"Prediction : {class_name}")
    print(f"Confidence : {confidence:.4f}")
    print(f"Probabilities: NORMAL={probs[0]:.4f}, PNEUMONIA={probs[1]:.4f}")


if __name__ == "__main__":
    main()
