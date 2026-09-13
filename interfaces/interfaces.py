from pathlib import Path
import logging
from util.config import AUDIO_TRANSCRIBE_PIPELINE, AUDIO_TRANSCRIBE_MODEL, LANGUAGE_MODEL, MODELS_DIR, MAX_NEW_TOKENS
from interfaces.pipelines import audio_pipeline, llm_pipeline
from util.prompts import language_model_prompt
from util.utils import markdown_to_pdf

logger = logging.getLogger(__name__)

def run_audio_pipeline(audio_file) -> str:
    """
    Runs audio transcipton pipeline and returns transcription text
    """
    logger.info("Processing audio transcription pipeline")
    
    transcription = audio_pipeline(audio_file=audio_file, 
                                   model=f"./{MODELS_DIR}/{AUDIO_TRANSCRIBE_MODEL}", 
                                   pipeline_task=AUDIO_TRANSCRIBE_PIPELINE
                                   )
    return transcription
    
def run_summarizer_pipeline(transcription) -> tuple[str,bytes]:
    """
    Runs the LLM pipeline and returns the response in a pdf file format
    """
    logger.info("Proceeding to LLM summarisation pipeline")
    
    if not transcription:
        logger.error("Transcription not found process discarded")
    else:  
        messages = language_model_prompt(transcription=transcription)
        
        logger.info("Transcription found, prompt completed")  
        logger.info("Sending prompt to LLM") 
        
        finalized_minutes = llm_pipeline(model=f"./{MODELS_DIR}/{LANGUAGE_MODEL}",
                                        messages=messages,
                                        max_new_output_tokens=MAX_NEW_TOKENS
                                        )
        try:
            if not finalized_minutes:
                logger.error("Missing LLM response, absorting operation")
            else:
                logger.info("Proceding to PDF conversion")
                pdf = markdown_to_pdf(finalized_minutes)
                
                if not pdf:
                    logger.error("Missing PDF, absorting operation")
                else:
                    logger.info("PDF generation complete")
                    return finalized_minutes, pdf
        except RuntimeError:
            logger.exception("Minutes finalization failed")
            raise RuntimeError("Minutes fininizaton errored")