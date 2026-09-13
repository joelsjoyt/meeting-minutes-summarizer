import logging
import sys
import gc
import torch
from .errors import DeviceError, LLMResponseCleanerError


logger = logging.getLogger(__name__)

    
def clear_gpu_cache():
    logger.info("Cleaning GPU cache")
    
    try:
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        logger.info(
            "GPU allocated: %.2f GB",
            torch.cuda.memory_allocated() / 1024**3,
        )

        logger.info(
            "GPU reserved: %.2f GB",
            torch.cuda.memory_reserved() / 1024**3,
        )
        
    except Exception:
        logger.warning("Failed to clear GPU cache", exc_info=True)
    
    
def get_device() -> str:
    """Return CUDA device or raise an error if unavailable."""
    if not torch.cuda.is_available():
        raise DeviceError(
            "CUDA is required to run this application. "
            "No CUDA-compatible GPU was detected."
        )

    return "cuda"


def setup_logger(root) -> None:
    """
    Initialize logger
    """
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.handlers.clear()
    root.addHandler(handler)
    
    
def clean_response(response):
    
    logger.info("Cleaning LLM response")
    
    if not response:
        raise LLMResponseCleanerError("Response is empty")
    
    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    response = response.strip()

    # Remove markdown code fence if model added one
    if response.startswith("```markdown"):
        response = response[len("```markdown"):].strip()
    elif response.startswith("```"):
        response = response[3:].strip()

    if response.endswith("```"):
        response = response[:-3].strip()

    return response

        
# def load_audio_file(path):
#     """
#     Loads audio file supplied via Gradio
#     """
#     print("Loading audio file")
#     audio_file = open(path, "rb")
#     return audio_file


    