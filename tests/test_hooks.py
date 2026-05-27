import torch

from src.utils.hooks import layer_from_fraction, mean_activation_after_token, residual_hook_name


def test_layer_from_fraction_selects_valid_layer():
    assert layer_from_fraction(32, 0.66) == 20
    assert layer_from_fraction(32, 0.0) == 0
    assert layer_from_fraction(32, 1.0) == 31


def test_residual_hook_name():
    assert residual_hook_name(20) == "blocks.20.hook_resid_post"


def test_mean_activation_after_token_handles_short_sequence():
    acts = torch.ones(2, 4, 3)
    result = mean_activation_after_token(acts, start_token=50)
    assert result.shape == (2, 3)
