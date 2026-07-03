from .config import *
from .model import *
from .tokenizer import *

import os
USE_NPU = os.environ.get("USE_NPU") == "True"
try:
    import torch_npu
    USE_NPU = torch.npu.is_available()
except ImportError:
    pass

if USE_NPU:
    import sys
    import olmo.memory_parallel_npu
    sys.modules['olmo.memory_parallel'] = olmo.memory_parallel_npu


def check_install(cuda: bool = False):
    import torch

    from .version import VERSION

    if cuda:
        assert torch.cuda.is_available(), "CUDA is not available!"
        print("CUDA available")

    print(f"OLMo v{VERSION} installed")
