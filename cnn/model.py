import torch
import torch.nn as nn
from torchvision import models


# TODO: try ResNet50 backbone instead
def build_model(num_classes=2, freeze_base=True, dropout=0.3):
    """Build a ResNet18 model with a custom classification head.

    Args:
        num_classes: Number of output classes.
        freeze_base: If True, freeze all convolutional layers.
        dropout: Dropout probability before the final linear layer.

    Returns:
        A modified ResNet18 model ready for fine-tuning.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


def unfreeze_layers(model, num_layers=2):
    """Unfreeze the last N residual blocks for fine-tuning.

    Args:
        model: A ResNet model.
        num_layers: Number of layer groups to unfreeze from the end.
    """
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def load_model(model_path, num_classes=2, device="cpu"):
    """Load a saved model from disk.

    Args:
        model_path: Path to the .pth file.
        num_classes: Number of output classes.
        device: Device to load the model onto.

    Returns:
        The loaded model in eval mode.
    """
    model = build_model(num_classes=num_classes, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
