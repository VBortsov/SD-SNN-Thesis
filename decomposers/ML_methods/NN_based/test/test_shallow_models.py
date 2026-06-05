import torch

from decomposers.ML_methods.NN_based.models_shallow.conv1Dnetwork import ShallowConv1DDecomposer
from decomposers.ML_methods.NN_based.models_shallow.mlp.singlehead import SingleHeadMLPDecomposer
from decomposers.ML_methods.NN_based.models_shallow.fuse import Fuse
from decomposers.ML_methods.NN_based.models_shallow.multiscale_branches import (
    AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer,
    AttentionStemMultiHeadMultiScaleTCNDecomposer,
    AttentionStemMultiHeadMultiScaleTCNDecomposerConfig,
    AttentionStemMultipleHeadMultiScaleBranchesDecomposer,
    BilinearFusion,
    MultiScaleBranchesDecomposer,
    MultipleHeadMultiScaleBranchesDecomposer,
)
from decomposers.ML_methods.NN_based.models_shallow.multiscale_dilated import ShallowMultiScaleDilatedDecomposer


def test_shallow_conv1d_forward_and_reconstruction_num_layers_1() -> None:
    model = ShallowConv1DDecomposer(in_channels=1, out_channels=3, hidden_channels=16, kernel_size=5, num_layers=1)
    x = torch.randn(4, 1, 257)
    y = model(x)
    assert y.shape == (4, 3, 257)
    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (4, 1, 257)


def test_shallow_conv1d_forward_and_reconstruction_num_layers_2() -> None:
    model = ShallowConv1DDecomposer(in_channels=1, out_channels=3, hidden_channels=16, kernel_size=3, num_layers=2)
    x = torch.randn(2, 1, 128)
    y = model(x)
    assert y.shape == (2, 3, 128)
    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (2, 1, 128)


def test_shallow_conv1d_depth_expansion_preserves_shape_and_backward() -> None:
    model = ShallowConv1DDecomposer(
        in_channels=1,
        out_channels=3,
        hidden_channels=16,
        kernel_size=5,
        num_layers=2,
        extra_conv_layers=3,
        extra_conv_kernel_size=3,
        extra_conv_channels=24,
        extra_conv_dilation=2,
        extra_conv_activation="gelu",
        extra_conv_norm="groupnorm",
        extra_conv_dropout=0.0,
        extra_conv_num_groups=8,
        extra_conv_residual=True,
    )
    x = torch.randn(2, 1, 257)
    target = torch.randn(2, 3, 257)
    y = model(x)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    assert y.shape == (2, 3, 257)
    assert model.depth_expansion.output_channels == 24
    assert any(parameter.grad is not None for parameter in model.depth_expansion.parameters())


def test_single_head_mlp_forward_and_reconstruction() -> None:
    model = SingleHeadMLPDecomposer(in_channels=1, out_channels=3, hidden_dim=32, activation="relu", dropout=0.1)
    x = torch.randn(3, 1, 200)
    y = model(x)
    assert y.shape == (3, 3, 200)
    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (3, 1, 200)


def test_model_registry_supports_shallow_models() -> None:
    from decomposers.ML_methods.NN_based.models import create_model

    conv = create_model("conv1dnetwork", in_channels=1, out_channels=3)
    conv_depth = create_model("conv1dnetwork", in_channels=1, out_channels=3, extra_conv_layers=2, extra_conv_channels=20)
    mlp = create_model("mlp_singlehead", in_channels=1, out_channels=3)
    multiscale_dilated = create_model("multiscale_dilated", in_channels=1, out_channels=3)
    multiple_head_multiscale = create_model("multiple_head_multiscale_branches", in_channels=1, out_channels=3)
    attention_stem_multiscale = create_model(
        "attention_stem_multiple_head_multiscale_branches",
        in_channels=1,
        out_channels=3,
    )
    bilinear_attention_stem_multiscale = create_model(
        "attention_stem_bilinear_fusion_multiple_head_multiscale_branches",
        in_channels=1,
        out_channels=3,
    )
    attention_stem_tcn = create_model(
        "attention_stem_multi_head_multiscale_tcn",
        in_channels=1,
        out_channels=3,
    )

    assert conv.__class__.__name__ == "ShallowConv1DDecomposer"
    assert conv_depth.depth_expansion.output_channels == 20
    assert mlp.__class__.__name__ == "SingleHeadMLPDecomposer"
    assert multiscale_dilated.__class__.__name__ == "ShallowMultiScaleDilatedDecomposer"
    assert multiple_head_multiscale.__class__.__name__ == "MultipleHeadMultiScaleBranchesDecomposer"
    assert attention_stem_multiscale.__class__.__name__ == "AttentionStemMultipleHeadMultiScaleBranchesDecomposer"
    assert attention_stem_tcn.__class__.__name__ == "AttentionStemMultiHeadMultiScaleTCNDecomposer"
    assert (
        bilinear_attention_stem_multiscale.__class__.__name__
        == "AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer"
    )


def test_fuse_forward_preserves_length_and_reduces_channels() -> None:
    model = Fuse(in_channels=96, out_channels=48, norm="groupnorm", activation="gelu", dropout=0.0, num_groups=8)
    x = torch.randn(5, 96, 257)
    y = model(x)
    assert y.shape == (5, 48, 257)


def test_fuse_rejects_wrong_channel_count() -> None:
    model = Fuse(in_channels=96, out_channels=48)
    x = torch.randn(2, 95, 64)
    try:
        _ = model(x)
    except ValueError as exc:
        assert "Expected x.shape[1]" in str(exc)
    else:
        raise AssertionError("Fuse should raise ValueError when channel count is wrong.")


def test_multiscale_branches_with_fusion_forward_and_reconstruction() -> None:
    model = MultiScaleBranchesDecomposer(
        in_channels=1,
        out_channels=3,
        branch_channels=16,
        fused_channels=24,
        dropout=0.0,
        num_groups=8,
    )
    x = torch.randn(3, 1, 211)
    y = model(x)
    assert y.shape == (3, 3, 211)
    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (3, 1, 211)


def test_multiscale_branches_depth_expansion_preserves_shape() -> None:
    model = MultiScaleBranchesDecomposer(
        in_channels=1,
        out_channels=3,
        branch_channels=16,
        fused_channels=24,
        dropout=0.0,
        num_groups=8,
        extra_conv_layers=2,
        extra_conv_kernel_size=5,
        extra_conv_channels=32,
        extra_conv_dilation=2,
        extra_conv_activation="gelu",
        extra_conv_norm="groupnorm",
        extra_conv_dropout=0.0,
        extra_conv_residual=True,
    )
    x = torch.randn(2, 1, 211)
    y = model(x)
    assert y.shape == (2, 3, 211)
    assert model.depth_expansion.output_channels == 32


def test_multiple_head_multiscale_branches_forward_and_reconstruction() -> None:
    model = MultipleHeadMultiScaleBranchesDecomposer(
        in_channels=1,
        out_channels=4,
        branch_channels=16,
        fused_channels=24,
        dropout=0.0,
        num_groups=8,
    )
    x = torch.randn(3, 1, 211)
    y = model(x)
    assert y.shape == (3, 4, 211)
    assert len(model.heads) == 4
    assert all(head.out_channels == 1 for head in model.heads)
    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (3, 1, 211)


def test_attention_stem_multiple_head_multiscale_branches_forward_attention_and_reconstruction() -> None:
    model = AttentionStemMultipleHeadMultiScaleBranchesDecomposer(
        in_channels=1,
        out_channels=4,
        stem_channels=16,
        branch_channels=16,
        fused_channels=24,
        attention_hidden_channels=12,
        dropout=0.0,
        num_groups=8,
    )
    x = torch.randn(3, 1, 211)
    y = model(x)
    assert y.shape == (3, 4, 211)
    assert len(model.heads) == 4
    assert all(head.out_channels == 1 for head in model.heads)

    branch_features = model.features(model.stem(x))
    weights = model.branch_attention_weights(branch_features)
    assert weights.shape == (3, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)

    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (3, 1, 211)


def test_bilinear_fusion_forward_shape() -> None:
    model = BilinearFusion(
        in_channels=48,
        out_channels=24,
        bilinear_channels=32,
        norm="groupnorm",
        activation="gelu",
        dropout=0.0,
        num_groups=8,
    )
    x = torch.randn(3, 48, 211)
    y = model(x)
    assert y.shape == (3, 24, 211)


def test_attention_stem_bilinear_fusion_multiple_head_multiscale_branches_forward_attention_and_reconstruction() -> None:
    model = AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer(
        in_channels=1,
        out_channels=4,
        stem_channels=16,
        branch_channels=16,
        fused_channels=24,
        attention_hidden_channels=12,
        bilinear_channels=32,
        dropout=0.0,
        num_groups=8,
    )
    x = torch.randn(3, 1, 211)
    y = model(x)
    assert y.shape == (3, 4, 211)
    assert len(model.heads) == 4
    assert all(head.out_channels == 1 for head in model.heads)

    branch_features = model.features(model.stem(x))
    weights = model.branch_attention_weights(branch_features)
    assert weights.shape == (3, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)

    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (3, 1, 211)


def test_attention_stem_multi_head_multiscale_tcn_config_instantiates() -> None:
    cfg = AttentionStemMultiHeadMultiScaleTCNDecomposerConfig()
    assert cfg.in_channels == 1
    assert cfg.out_channels == 3
    assert cfg.tcn_dilations == (1, 2, 4, 8)


def test_attention_stem_multi_head_multiscale_tcn_forward_reconstruction_and_attention() -> None:
    model = AttentionStemMultiHeadMultiScaleTCNDecomposer()
    x = torch.randn(2, 1, 1024)
    y = model(x)
    assert y.shape == (2, 3, 1024)

    stem_features = model.stem(model._augment_input(x))
    weights = model.branch_attention_weights(stem_features)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)

    reconstructed = model.reconstruction(y)
    assert reconstructed.shape == (2, 1, 1024)


def test_attention_stem_multi_head_multiscale_tcn_frequency_features_preserve_shape() -> None:
    model = AttentionStemMultiHeadMultiScaleTCNDecomposer(
        use_frequency_features=True,
        frequency_feature_mode="fft_magnitude",
    )
    x = torch.randn(2, 1, 1024)
    y = model(x)
    assert y.shape == (2, 3, 1024)


def test_attention_stem_multi_head_multiscale_tcn_causal_mode_preserves_shape() -> None:
    model = AttentionStemMultiHeadMultiScaleTCNDecomposer(causal=True)
    x = torch.randn(2, 1, 1024)
    y = model(x)
    assert y.shape == (2, 3, 1024)


def test_attention_stem_multi_head_multiscale_tcn_backward_and_parameter_budget() -> None:
    model = AttentionStemMultiHeadMultiScaleTCNDecomposer()
    x = torch.randn(2, 1, 1024)
    target = torch.randn(2, 3, 1024)
    y = model(x)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    assert model.stem[0].conv.weight.grad is not None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 250_000


def test_decomposition_loss_supports_optional_component_specific_terms() -> None:
    from types import SimpleNamespace

    from decomposers.ML_methods.NN_based.training_common import build_decomposition_loss

    cfg = SimpleNamespace(
        component_loss_weight=1.0,
        reconstruction_loss_weight=0.5,
        spectral_loss_weight=0.05,
        component_loss_weights=(1.0, 0.5, 1.5),
        mixing_penalty_weight=0.1,
        frequency_band_loss_weight=0.1,
        frequency_band_ranges=((0.0, 20.0), (20.0, 40.0), (40.0, 80.0)),
        fs=256,
    )
    criterion = build_decomposition_loss(cfg)
    pred = torch.randn(2, 3, 1024, requires_grad=True)
    target = torch.randn(2, 3, 1024)
    mixture = target.sum(dim=1, keepdim=True)

    loss = criterion(pred, target, mixture)
    loss.backward()

    assert loss.ndim == 0
    assert pred.grad is not None


def test_multiscale_dilated_stem_preserves_shape() -> None:
    hidden_channels = 32
    model = ShallowMultiScaleDilatedDecomposer(hidden_channels=hidden_channels, stem_kernel_size=9)
    x = torch.randn(4, 1, 257)
    features = model.stem(x)
    assert features.shape == (4, hidden_channels, 257)


def test_multiscale_dilated_forward_shape() -> None:
    model = ShallowMultiScaleDilatedDecomposer(in_channels=1, out_channels=3, hidden_channels=16, stem_kernel_size=9)
    x = torch.randn(3, 1, 211)
    y = model(x)
    assert y.shape == (3, 3, 211)


def test_multiscale_dilated_rejects_even_stem_kernel_size() -> None:
    try:
        _ = ShallowMultiScaleDilatedDecomposer(stem_kernel_size=8)
    except ValueError as exc:
        assert "positive odd integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for even stem_kernel_size.")


def test_multiscale_dilated_activation_options_construct() -> None:
    for activation in ("relu", "gelu", "tanh"):
        model = ShallowMultiScaleDilatedDecomposer(activation=activation)
        assert isinstance(model, ShallowMultiScaleDilatedDecomposer)
