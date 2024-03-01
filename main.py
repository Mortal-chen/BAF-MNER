from config import Config
import torch
import numpy as np
from frameworks.framework import Framework

if __name__ == '__main__':
    seed = 2023
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config()
    fw = Framework(config)
    fw.train()
