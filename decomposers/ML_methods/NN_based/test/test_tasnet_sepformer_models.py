import torch

from decomposers.ML_methods.NN_based.models import create_model
from decomposers.ML_methods.NN_based.models.sepformer import SepFormerDecomposer
from decomposers.ML_methods.NN_based.models.tasnet import TasNetDecomposer


def test_tasnet_forward_shape_and_reconstruction() -> None:
    model = TasNetDecomposer(
        in_channels=1,
        out_channels=3,
        encoder_dim=64,
        bottleneck_dim=64,
        hidden_dim=128,
        skip_dim=64,
        kernel_size=8,
        stride=4,
        num_blocks=3,
        num_repeats=2,
        mask_activation="sigmoid",
    )
    x = torch.randn(2, 1, 257)
    y = model(x)
    assert y.shape == (2, 3, 257)

    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (2, 1, 257)


def test_sepformer_forward_shape_and_reconstruction() -> None:
    model = SepFormerDecomposer(
        in_channels=1,
        out_channels=3,
        encoder_dim=64,
        bottleneck_dim=64,
        kernel_size=8,
        stride=4,
        chunk_size=24,
        hop_size=12,
        num_sepformer_blocks=1,
        num_attention_heads=8,
        feedforward_dim=128,
    )
    x = torch.randn(2, 1, 255)
    y = model(x)
    assert y.shape == (2, 3, 255)

    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (2, 1, 255)


def test_model_registry_supports_tasnet_and_sepformer() -> None:
    tasnet = create_model("tasnet", in_channels=1, out_channels=3, encoder_dim=32, bottleneck_dim=32, hidden_dim=64, skip_dim=32, num_blocks=2, num_repeats=1)
    sepformer = create_model("sepformer", in_channels=1, out_channels=3, encoder_dim=32, bottleneck_dim=32, chunk_size=16, hop_size=8, num_sepformer_blocks=1, num_attention_heads=8, feedforward_dim=64)

    assert tasnet.__class__.__name__ == "TasNetDecomposer"
    assert sepformer.__class__.__name__ == "SepFormerDecomposer"
