import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """固定随机种子，便于复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
