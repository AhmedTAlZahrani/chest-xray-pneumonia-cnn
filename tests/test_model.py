import pytest
import torch
import torch.nn as nn
import numpy as np

from cnn.model import build_model, unfreeze_layers, load_model


# -- build_model tests --

def test_build_model_returns_resnet():
    """Default build gives a ResNet with 2-class head."""
    model = build_model()
    assert isinstance(model.fc, nn.Sequential)
    # final linear should output 2 classes
    linear = model.fc[1]
    assert linear.out_features == 2


def test_build_model_custom_classes():
    """Num classes propagates to the output layer."""
    model = build_model(num_classes=5, freeze_base=False)
    linear = model.fc[1]
    assert linear.out_features == 5


def test_build_model_freeze_base():
    """Frozen base means conv params have requires_grad=False."""
    model = build_model(freeze_base=True)
    for name, param in model.named_parameters():
        if "fc" not in name:
            assert not param.requires_grad, f"{name} should be frozen"


def test_build_model_unfreeze_base():
    """Unfrozen base means all params are trainable."""
    model = build_model(freeze_base=False)
    for param in model.parameters():
        assert param.requires_grad


def test_build_model_dropout():
    """Dropout layer uses the specified probability."""
    model = build_model(dropout=0.5)
    dropout_layer = model.fc[0]
    assert isinstance(dropout_layer, nn.Dropout)
    assert dropout_layer.p == 0.5


# -- forward pass --

def test_forward_pass_shape():
    """Output shape should be (batch, num_classes)."""
    model = build_model(num_classes=2, freeze_base=True)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2)


def test_forward_single_image():
    """Single image forward pass."""
    model = build_model(num_classes=3)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3)


def test_forward_output_not_all_same():
    """Different inputs should usually produce different outputs."""
    model = build_model()
    model.eval()
    a = torch.randn(1, 3, 224, 224)
    b = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        oa = model(a)
        ob = model(b)
    assert not torch.allclose(oa, ob)


# -- unfreeze_layers --

def test_unfreeze_layers_default():
    """Unfreezing 2 layers makes layer3 and layer4 trainable."""
    model = build_model(freeze_base=True)
    unfreeze_layers(model, num_layers=2)
    for param in model.layer3.parameters():
        assert param.requires_grad
    for param in model.layer4.parameters():
        assert param.requires_grad
    # layer1 should stay frozen
    for param in model.layer1.parameters():
        assert not param.requires_grad


def test_unfreeze_single_layer():
    """Unfreezing 1 layer only touches layer4."""
    model = build_model(freeze_base=True)
    unfreeze_layers(model, num_layers=1)
    for param in model.layer4.parameters():
        assert param.requires_grad
    for param in model.layer3.parameters():
        assert not param.requires_grad


# -- load_model --

def test_load_model_roundtrip(tmp_path):
    """Save then load produces identical predictions."""
    model = build_model(num_classes=2, freeze_base=False)
    model.eval()
    save_path = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), save_path)

    loaded = load_model(str(save_path), num_classes=2, device="cpu")
    assert not loaded.training  # should be in eval mode

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        orig_out = model(x)
        loaded_out = loaded(x)
    assert torch.allclose(orig_out, loaded_out, atol=1e-6)


def test_load_model_eval_mode(tmp_path):
    """Loaded model is in eval mode."""
    model = build_model()
    torch.save(model.state_dict(), tmp_path / "m.pth")
    loaded = load_model(str(tmp_path / "m.pth"))
    assert not loaded.training


# -- softmax sanity --

def test_softmax_probabilities():
    """Softmax of model output sums to 1."""
    model = build_model()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


# -- GPU tests --

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_pass_gpu():
    """Forward pass on GPU produces correct shape."""
    device = torch.device("cuda")
    model = build_model().to(device)
    model.eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
    assert out.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_model_gpu(tmp_path):
    """Load model onto GPU."""
    model = build_model()
    torch.save(model.state_dict(), tmp_path / "gpu.pth")
    loaded = load_model(str(tmp_path / "gpu.pth"), device="cuda")
    assert next(loaded.parameters()).device.type == "cuda"
