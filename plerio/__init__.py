def seed_everything(seed: int = 43):
    """
    Allows to fix all the randomness.
    """
    # OS-level deterministic config
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import numpy as np
    import random
    import torch.backends.cudnn as cudnn

    # Seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Torch deterministic configs
    torch.use_deterministic_algorithms(True)
    cudnn.deterministic = True
    cudnn.benchmark = False
