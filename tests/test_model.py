"""Tests for the CRNN model and masked loss."""

import torch

from pytorch_core.models.crnn import CRNN, CustomBCELoss


def test_forward_output_shape(config):
    """The model maps [batch, meters, frames, features] to [batch, meters, 1]."""
    model = CRNN(config)
    batch, meters = 2, config["data"]["max_meters"]
    frames, feats = config["data"]["max_frames"], config["data"]["n_features"]

    x = torch.randn(batch, meters, frames, feats)
    out = model(x)

    assert out.shape == (batch, meters, 1)
    assert torch.all((out >= 0) & (out <= 1))  # sigmoid output


def test_cnn_output_dim_matches_tf(config):
    """Flattened CNN dimension matches the original Keras model (38 * 256)."""
    model = CRNN(config)
    assert model.cnn_output_dim == 9728


def test_padded_meters_are_zeroed(config):
    """Meters that are entirely zero (padding) produce a zero prediction."""
    model = CRNN(config)
    model.eval()
    meters = config["data"]["max_meters"]
    frames, feats = config["data"]["max_frames"], config["data"]["n_features"]

    x = torch.zeros(1, meters, frames, feats)
    x[0, :5] = torch.randn(5, frames, feats)  # only first 5 meters have content

    with torch.no_grad():
        out = model(x)

    assert torch.all(out[0, 5:] == 0)  # padded meters masked to zero


def test_masked_loss_ignores_padding():
    """Loss is unchanged by the value of padded (-1) targets."""
    criterion = CustomBCELoss()
    preds = torch.tensor([[[0.9], [0.1], [0.5]]])
    targets_a = torch.tensor([[[1.0], [0.0], [-1.0]]])
    targets_b = torch.tensor([[[1.0], [0.0], [-1.0]]])

    # Change the prediction under the padded position only
    preds_shifted = preds.clone()
    preds_shifted[0, 2, 0] = 0.999

    loss_a = criterion(preds, targets_a)
    loss_b = criterion(preds_shifted, targets_b)

    assert torch.isclose(loss_a, loss_b)


def test_masked_loss_all_padding_is_safe():
    """A batch that is entirely padding does not produce NaN."""
    criterion = CustomBCELoss()
    preds = torch.tensor([[[0.5], [0.5]]])
    targets = torch.tensor([[[-1.0], [-1.0]]])

    loss = criterion(preds, targets)
    assert not torch.isnan(loss)


def test_loss_backward_runs(config):
    """A full forward/backward pass produces finite gradients."""
    model = CRNN(config)
    criterion = CustomBCELoss()

    x = torch.randn(2, config["data"]["max_meters"],
                    config["data"]["max_frames"], config["data"]["n_features"])
    targets = torch.full((2, config["data"]["max_meters"], 1), -1.0)
    targets[:, :50] = torch.randint(0, 2, (2, 50, 1)).float()

    loss = criterion(model(x), targets)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
