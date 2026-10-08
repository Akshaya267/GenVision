import torch
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenVision.ModelUtils")

def get_device_and_dtype():
    """
    Detects available hardware acceleration (CUDA GPU vs CPU).
    Returns (device_str, torch_dtype).
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA Hardware Accelerator Detected: {gpu_name} (Using float16)")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("CUDA GPU not available. Falling back to CPU execution (Using float32)")
    return device, dtype

def clear_memory():
    """
    Forces garbage collection and releases cached VRAM to prevent OOM errors.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("CUDA VRAM cache flushed.")
