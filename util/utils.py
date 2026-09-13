import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
import markdown2
from weasyprint import HTML
import tempfile
from util.config import MODELS_DIR, AUDIO_TRANSCRIBE_MODEL, LANGUAGE_MODEL
import gc
import torch


def login_HF() -> None:
    """
    Login to Huggingface Hub to download gated or private models
    """
    
    load_dotenv(override=True)
    hf_token = os.getenv('HF_TOKEN')
    login(hf_token, add_to_git_credential=True)
    
def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    
    
def download_models(model_repo) -> str:
    """ 
    Use login_HF when model is gated or private
    This function downloads models if it does not exist
    """
    
    # login_HF()
    model_dir = Path(MODELS_DIR) / model_repo
    
    # Split model_repo into organization and model name
    repo_parts = model_repo.split("/", 1)

    if len(repo_parts) != 2:
        raise ValueError(
            f"Invalid Hugging Face repo ID: {model_repo}. "
            "Expected format: 'organization/model'"
        )

    organization, model_name = repo_parts
    
    organization_dir = Path(MODELS_DIR) / organization
    model_dir = organization_dir / model_name
    
    # Check exact path:
    # models/openai/whisper-medium.en
    if organization_dir.is_dir() and model_dir.is_dir():
        print(f"✓ Model already exists: {model_repo}")
        return str(model_dir)
    
    print(f"↓ Downloading: {model_repo}")

    # Creates:
    # models/
    # └── repo/
    #     └── model-name/
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_repo,
        local_dir=str(model_dir),
    )

    return str(model_dir)
    

def download_required_models() -> bool:
    """
    This function downloads required models for this app
    """
    
    audio_model_path = download_models(AUDIO_TRANSCRIBE_MODEL)
    language_model_path = download_models(LANGUAGE_MODEL)
    
    if audio_model_path and language_model_path:
        return True
    else:
        return False
    
# def load_audio_file(path):
#     """
#     Loads audio file supplied via Gradio
#     """
#     print("Loading audio file")
#     audio_file = open(path, "rb")
#     return audio_file


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
    
    
def markdown_to_pdf(md):
    """
    This converts markdown response to pdf to be downloaded
    """
    
    html = markdown2.markdown(
        md,
        extras=["tables", "fenced-code-blocks"]
    )

    pdf_file = tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    )

    HTML(string=html).write_pdf(pdf_file.name)

    return pdf_file.name

def clean_response(response):
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


    