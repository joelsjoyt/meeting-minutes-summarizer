import torch
import logging
from util.utils import clear_gpu_cache, clean_response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, pipeline

logger = logging.getLogger(__name__)

def quantizer():
    """"
    This function applies 4 Bit quantization to the supplied model
    """
    quantizer_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )
    
    logger.info("Quantizer config loaded")
    
    return quantizer_config

def audio_pipeline(model, audio_file, pipeline_task) -> str:
    """"
    This function contains the pipeline to create a text transcription from audio file
    """
    logger.info("Configuring audio transcription pipeline")
    pipe = pipeline(
        pipeline_task,
        model=model,
        dtype=torch.float16,
        device='cuda',
        return_timestamps=True
    )

    result = pipe(audio_file)
    if not result:
        logger.error("Transcription failed")
    else:
        transcription = result["text"]
        logger.info("Transcription succesful")
    
    del pipe
    clear_gpu_cache()
    
    return transcription
    
def llm_pipeline(model, messages, max_new_output_tokens) -> str:
    """"
    This function takes in a transcription as text and returns a textual response
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    if not tokenizer: 
        logger.error("Autotokenizer failed")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                           add_generation_prompt=True, 
                                           return_dict=False,
                                           ).to("cuda")   
    input_length = inputs.shape[1]
    # streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quantizer())
    logger.info("LLM model loaded")
    outputs = model.generate(inputs, max_new_tokens=max_new_output_tokens)
    
    if outputs.numel() == 0:
        logger.error("LLM outputs are empty")
    logger.info("LLM output completed")
    
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
    del inputs
    del tokenizer
    del model
    del outputs
    clear_gpu_cache()
    
    response = clean_response(response=response)
    
    logger.info("LLM cleaned response is retrived")
    
    return response

    
