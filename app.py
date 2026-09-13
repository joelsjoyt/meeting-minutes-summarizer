""""
This file acts as the entrypoint for the app
"""

import logging
import torch
from util.utils import setup_logger, get_device
from util.downloads import download_required_models
from views.app_view import run_app_ui
from util.errors import ApplicationError, ModelDownloadPipelineError, DeviceError


def main():
    root = logging.getLogger()
    setup_logger(root)
    
    logger = logging.getLogger(__name__)
    logger.info("Application has started")  
    
    try:
        
        DEVICE = get_device()
            
        logger.info(f"Using device: {DEVICE}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")  
        
        download_required_models()
        run_app_ui()
        
    except DeviceError as e:
        logger.exception("Encountered device exception")
        raise ApplicationError(
            "Encountered device exception"
        ) from e
        
    except ModelDownloadPipelineError as e:
        logger.exception("Model download pipeline exception")
        raise ApplicationError(
            "Model download pipeline error"
        ) from e
        
    except Exception as e:
        raise ApplicationError(
            "Application runtime error"
        ) from e
    
if __name__ == "__main__":
    main()
    





    
