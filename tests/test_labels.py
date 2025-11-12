import torch

from src.utils.loss import make_next_token_labels


def test_make_next_token_labels_shifts_and_masks():
    tokens = torch.tensor([[5, 6, 7, 0]])
    labels = make_next_token_labels(tokens, pad_id=0)
    assert labels.shape == tokens.shape
    assert labels[0, 0].item() == 6
    assert labels[0, 1].item() == 7
    # PAD positions and last position should be ignored
    assert labels[0, 2].item() == 0
    assert labels[0, 3].item() == -100


def test_make_next_token_labels_single_token():
    tokens = torch.tensor([[9]])
    labels = make_next_token_labels(tokens, pad_id=0)
    assert labels[0, 0].item() == -100
