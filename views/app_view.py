import gradio as gr
import logging
import os
from dotenv import load_dotenv
from util.styles import gradio_style
from interfaces.interfaces import run_audio_pipeline, run_summarizer_pipeline

logger = logging.getLogger(__name__)

def run_app_ui():
    """
    Gradio UI
    """
    load_dotenv(override=True)
    
    with gr.Blocks(css=gradio_style) as ui:
        with gr.Column(elem_classes="main-container"):
            with gr.Row(equal_height=True, elem_classes="section"):
                with gr.Column(scale=1):
                    audio = gr.UploadButton(label="🎵 Upload Audio",
                                            file_types=["audio"],
                                            min_width=100
                                            )
                    pdf = gr.DownloadButton("📄 Download PDF", 
                                        elem_classes="download-btn")
                with gr.Column(scale=8):
                    transcribed_audio = gr.Textbox(label="Transcribed Audio",
                                                lines=18,
                                                max_lines=20
                                                )
            with gr.Row(equal_height=True, elem_classes="section"):
                with gr.Column(scale=8, elem_classes="minutes-container"):
                    gr.Markdown("### 📝 Finalized Minutes", elem_classes="minutes-title")
                    finalized_minutes = gr.Markdown(elem_classes="minutes-content")
                                
                audio.upload(run_audio_pipeline, inputs=[audio], outputs=[transcribed_audio]).then(
                    run_summarizer_pipeline, inputs=[transcribed_audio], outputs=[finalized_minutes, pdf]
                )
        
    # ui.launch(inbrowser=True, auth=("app", "abc"))
    
    gradio_user_name = os.getenv('GRADIO_USER_NAME')
    gradio_user_pass = os.getenv('GRADIO_USER_PASS')
    
    if not gradio_user_name and not gradio_user_pass:
        logger.error("Restricted Access...")
    else:
        ui.launch(inbrowser=True, auth=(gradio_user_name, gradio_user_pass))