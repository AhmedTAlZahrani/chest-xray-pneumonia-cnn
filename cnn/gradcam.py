import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import get_transforms
from .model import load_model


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            Numpy array heatmap normalized to [0, 1].
        """
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(image_path, heatmap, alpha=0.4):
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        image_path: Path to the original image.
        heatmap: 2D numpy heatmap array.
        alpha: Blending factor for the overlay.

    Returns:
        BGR image with heatmap overlay.
    """
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/best_model.pth")
    parser.add_argument("--output_path", type=str, default="output/gradcam.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device=device)

    # Hook into the last convolutional layer of ResNet18
    target_layer = model.layer4[1].conv2
    gradcam = GradCAM(model, target_layer)

    transform = get_transforms(split="test")
    image = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    heatmap = gradcam.generate(input_tensor)
    result = overlay_heatmap(args.image_path, heatmap)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"Grad-CAM heatmap saved to {output_path}")


if __name__ == "__main__":
    main()
