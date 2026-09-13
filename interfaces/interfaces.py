from pathlib import Path
from util.config import AUDIO_TRANSCRIBE_PIPELINE, AUDIO_TRANSCRIBE_MODEL, LANGUAGE_MODEL, MODELS_DIR, MAX_NEW_TOKENS
from interfaces.pipelines import audio_pipeline, llm_pipeline
from util.prompts import language_model_prompt
from util.utils import markdown_to_pdf

def run_audio_pipeline(audio_file) -> str:
    """
    Runs audio transcipton pipeline and returns transcription text
    """
    print("Running audio tranbscribe pipeline")
    print(f"Audio file {audio_file}")
    transcription = audio_pipeline(audio_file=audio_file, 
                                   model=f"./{MODELS_DIR}/{AUDIO_TRANSCRIBE_MODEL}", 
                                   pipeline_task=AUDIO_TRANSCRIBE_PIPELINE
                                   )
    return transcription
    
def run_summarizer_pipeline(transcription) -> tuple[str,bytes]:
    """
    Runs the LLM pipeline and returns the response in a pdf file format
    """
    print("Running LLM pipeline")
    print(f"Transcribed audio: {transcription}")
    messages = language_model_prompt(transcription=transcription)
    print(f"Messages: {messages}")
    finalized_minutes = llm_pipeline(model=f"./{MODELS_DIR}/{LANGUAGE_MODEL}",
                                     messages=messages,
                                     max_new_output_tokens=MAX_NEW_TOKENS
                                     )
    pdf = markdown_to_pdf(finalized_minutes)
    return finalized_minutes, pdf