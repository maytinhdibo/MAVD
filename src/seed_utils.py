import os
import random

import numpy as np
import torch

def set_seed(seed: int = 42) -> None:

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id: int) -> None:

    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)

def make_generator(seed: int) -> torch.Generator:

    g = torch.Generator()
    g.manual_seed(seed)
    return g
