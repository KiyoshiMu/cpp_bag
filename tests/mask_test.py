from __future__ import annotations

import numpy as np
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


def test_mask_flag():
    mask_flag = np.zeros(5468, dtype=bool)
    mask_flag[np.random.choice(5468, size=1000, replace=False)] = True
    assert mask_flag.sum() == 1000
