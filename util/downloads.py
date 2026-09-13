import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from .errors import ModelDownloadError, ModelDownloadPipelineError
from util.config import MODELS_DIR, AUDIO_TRANSCRIBE_MODEL, LANGUAGE_MODEL

logger = logging.getLogger(__name__)

def login_HF() -> None:
    """
    Login to Huggingface Hub to download gated or private models
    """
    logger.info("Accessing Huggingface Hub")
    
    load_dotenv(override=True)
    hf_token = os.getenv('HF_TOKEN')
    login(hf_token, add_to_git_credential=True)

def download_models(model_repo) -> str:
    """ 
    Use login_HF when model is gated or private
    This function downloads models if it does not exist
    """
    
    logger.info(f"Checking model {model_repo}")
    
    try:
        # login_HF()
        model_dir = Path(MODELS_DIR) / model_repo
        
        # Split model_repo into organization and model name
        repo_parts = model_repo.split("/", 1)

        if len(repo_parts) != 2:
            raise ModelDownloadError(
                f"Invalid Hugging Face repo ID: {model_repo}. "
                "Expected format: 'organization/model'"
            )

        organization, model_name = repo_parts
        
        model_dir = Path(MODELS_DIR) / organization / model_name
        
        # Check exact path:
        # models/openai/whisper-medium.en
        if model_dir.is_dir():
            logger.info(f"✓ Model already exists: {model_repo}")
            return str(model_dir)
        
        logger.info(f"↓ Downloading: {model_repo}")

        # Creates:
        # models/
        # └── repo/
        #     └── model-name/
        model_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_repo,
            local_dir=str(model_dir),
        )
        
        if not model_dir.is_dir():
            raise ModelDownloadError(
                f"Model download completed but model directory "
                f"was not found: {model_repo}"
            )

        logger.info(f"✓ Model download complete: {model_repo}")
        return str(model_dir)
    
    except ModelDownloadError:
        raise
    
    except Exception as e:
        raise ModelDownloadError(
            f"Failed to download model: {model_repo}"
        ) from e
    

def download_required_models() -> bool:
    """
    This function downloads required models for this app
    """
    
    logger.info("Downloading required models")
    try:
        audio_model_path = download_models(AUDIO_TRANSCRIBE_MODEL)
        language_model_path = download_models(LANGUAGE_MODEL)
    
        if not audio_model_path:
            raise ModelDownloadPipelineError(
                "Audio model download failed"
            )
        
        if not language_model_path:
            raise ModelDownloadPipelineError(
                "LLM Model download failed"
            )
        
        return True
   
    except ModelDownloadPipelineError:
        pass
    
    except ModelDownloadError as e:
        raise ModelDownloadPipelineError(
            "Model download failed"
        ) from e
    
    except Exception as e:
        raise ModelDownloadPipelineError(
            "Unable to download model"
        ) from e