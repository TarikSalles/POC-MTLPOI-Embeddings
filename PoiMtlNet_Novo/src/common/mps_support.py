import gc

import torch


def clear_mps_cache():
    """Thoroughly clear MPS memory between training runs"""
    gc.collect()
    torch.mps.empty_cache()
    if hasattr(torch.mps, 'synchronize'):  # Check if available in your PyTorch version
        torch.mps.synchronize()  # Ensure all operations are complete