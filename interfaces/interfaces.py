from pathlib import Path
import logging
from util.config import (
                         AUDIO_TRANSCRIBE_PIPELINE, 
                         AUDIO_TRANSCRIBE_MODEL, 
                         LANGUAGE_MODEL, 
                         MODELS_DIR, 
                         MAX_NEW_TOKENS
                         )
from interfaces.pipelines import audio_pipeline, llm_pipeline
from util.prompts import language_model_prompt
from util.makrdown_to_pdf import markdown_to_pdf
from util.errors import (
                         TranscriptionPipelineError, 
                         SummarizationPipelineError, 
                         SummarizationModelError,
                         TranscriptionError,
                         PDFGenerationError
                         )

logger = logging.getLogger(__name__)

def run_audio_pipeline(audio_file) -> str:
    """
    Runs audio transcipton pipeline and returns transcription text
    """
    logger.info("Processing audio transcription pipeline")
    try:
        if not audio_file:
            raise TranscriptionPipelineError(
                "Audio file not found"
            )
        transcription = audio_pipeline(audio_file=audio_file, 
                                    model=f"./{MODELS_DIR}/{AUDIO_TRANSCRIBE_MODEL}", 
                                    pipeline_task=AUDIO_TRANSCRIBE_PIPELINE
                                    )
        
        if not transcription:
            raise TranscriptionPipelineError(
                "Model transcription result is empty"
            )
        
        logger.info("Audio transcription succesful")
        
        return transcription
        
    except TranscriptionPipelineError:
        raise
    
    except TranscriptionError as e:
        logging.exception("Audio transcription exception")
        raise TranscriptionPipelineError(
            "Audio transcription exception"
        ) from e
    
    except Exception as e:
        logger.exception("Unexpected audio pipeline failure")
        raise TranscriptionPipelineError(
            "Audio transcription pipeline failed"
        ) from e
    
def run_summarizer_pipeline(transcription) -> tuple[str,bytes]:
    """
    Runs the LLM pipeline and returns the response in a pdf file format
    """
    logger.info("Proceeding to LLM summarisation pipeline")
    
    try:
        
        if not transcription:
            raise SummarizationPipelineError(
                "Transcription is empty"
            )
             
        messages = language_model_prompt(transcription=transcription)
        
        if not messages:
            raise SummarizationPipelineError(
                "Messages to LLM is empty"
            )
        
        logger.info("Transcription found, prompt completed")  
        logger.info("Sending prompt to LLM") 
        
        finalized_minutes = llm_pipeline(llm_model=f"./{MODELS_DIR}/{LANGUAGE_MODEL}",
                                        messages=messages,
                                        max_new_output_tokens=MAX_NEW_TOKENS
                                        )
        
        if not finalized_minutes:
            raise SummarizationPipelineError(
                "LLM output is empty"
            )
        
        logger.info("Proceding to PDF conversion")
        pdf = markdown_to_pdf(finalized_minutes)
        
        if not pdf:
            raise SummarizationPipelineError(
                "PDF conversion failed"
            )
        
        logger.info("PDF generation complete")
        return finalized_minutes, pdf
       
    except SummarizationPipelineError:
        raise  
           
    except SummarizationModelError as e:
        logging.exception("Sumamrization model exception")
        raise SummarizationPipelineError(
            "Summarization model exception"
        ) from e      
        
    except PDFGenerationError as e:
        logging.exception("Summarization PDF generation failed")
        raise PDFGenerationError(
            "PDF generation failed"
        ) from e
    
    except Exception as e:
        logger.exception("Unexpected LLM pipeline failure")
        raise SummarizationPipelineError(
            "Summarization pipeline failed"
        ) from e
    