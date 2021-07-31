import torch
import pytest
from simple_trainer import helpers


def test_correct():
    outputs = torch.randn(32, 10)
    labels = torch.randint(10, (1, 32))
    assert int((outputs.argmax(1) == labels).sum()) == helpers.correct(outputs, labels)
