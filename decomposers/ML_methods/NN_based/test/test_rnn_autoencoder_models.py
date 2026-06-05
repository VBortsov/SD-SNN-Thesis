import torch

from decomposers.ML_methods.NN_based.models.autoencoderbased import AutoencoderDecomposer
from decomposers.ML_methods.NN_based.models.rnnbased import GRUDecomposer, RNNDecomposer


def test_rnn_decomposer_forward_shape_and_reconstruction() -> None:
    model = RNNDecomposer(
        in_channels=1,
        out_channels=3,
        hidden_size=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
        cell_type="lstm",
    )
    x = torch.randn(4, 1, 256)
    y = model(x)

    assert y.shape == (4, 3, 256)

    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (4, 1, 256)


def test_gru_variant_forward_shape() -> None:
    model = GRUDecomposer(in_channels=1, out_channels=3, hidden_size=32, num_layers=1, bidirectional=False)
    x = torch.randn(2, 1, 128)
    y = model(x)
    assert y.shape == (2, 3, 128)


def test_autoencoder_decomposer_forward_shape_and_reconstruction() -> None:
    model = AutoencoderDecomposer(
        in_channels=1,
        out_channels=3,
        hidden_channels=(16, 32),
        depth=2,
        kernel_size=5,
        dropout=0.1,
    )
    x = torch.randn(4, 1, 255)
    y = model(x)

    assert y.shape == (4, 3, 255)

    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (4, 1, 255)
