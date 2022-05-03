from __future__ import annotations

import torch


def test_mask_clone():
    tensor = torch.rand(5, 10)
    mask_tensor = torch.zeros(10)
    mask = torch.tensor([True, False, True, False, True])
    masked = tensor.clone()
    masked[mask] = mask_tensor
    print(masked)
    print(tensor)
    assert tensor.sum() != masked.sum()
