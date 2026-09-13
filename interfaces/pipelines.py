import torch
import logging
from util.utils import clear_gpu_cache, clean_response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, pipeline
from util.errors import TranscriptionError, SummarizationModelError, LLMResponseCleanerError

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
    
    pipe = None
    result = None
    
    try:
        if not audio_file:
            raise TranscriptionError("Audio file not found")
        
        pipe = pipeline(
            pipeline_task,
            model=model,
            dtype=torch.float16,
            device='cuda',
            return_timestamps=True
        )

        result = pipe(audio_file)
        
        if not result:
            raise TranscriptionError(
                "Transcription returned an empty result"
            )
        
        transcription = result["text"]
        
        if not transcription:
            raise TranscriptionError(
                "Transcription result constains no text"
            )
            
        logger.info("Transcription succesful")
        return transcription
    
    except TranscriptionError:
        raise
    
    except Exception as e:
        raise TranscriptionError(
            "Audio transcription failed"
        )
        
    finally:
        
        if result is not None:
            del result
        
        if pipe is not None:
            del pipe
            
        clear_gpu_cache()
    
    
    
def llm_pipeline(llm_model, messages, max_new_output_tokens) -> str:
    """"
    This function takes in a transcription as text and returns a textual response
    """
    
    tokenizer = None
    inputs = None
    model = None
    outputs = None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        
        if not tokenizer: 
            raise SummarizationModelError(
                "Tokenizer is empty"
            )
            
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                            add_generation_prompt=True, 
                                            return_dict=False,
                                            ).to("cuda")   
                
        if inputs is None or inputs.numel() == 0:
            raise SummarizationModelError(
                "LLM inputs are empty"
            )
        input_length = inputs.shape[1]
          
        # streamer = TextStreamer(tokenizer)
          
        model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto", quantization_config=quantizer())
        
        if model is None:
            raise SummarizationModelError(
                "LLM model failed to load"
            )
            
        logger.info("LLM model loaded")
        
        outputs = model.generate(inputs, max_new_tokens=max_new_output_tokens)
        
        if outputs is None or outputs.numel() == 0:
            raise SummarizationModelError(
                "LLM outputs are empty"
            )
            
        logger.info("LLM output completed")
        
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                
        if not response or not response.strip():
            raise SummarizationModelError(
                "LLM returned an empty response"
            )
            
        response = clean_response(response=response)
        
        if not response or not response.strip():
            raise SummarizationModelError(
                "LLM returned an empty response after cleaning"
            )

        logger.info("LLM cleaned response is retrived")
        
        return response
    
    except SummarizationModelError:
        raise
    
    except LLMResponseCleanerError as e:
        raise SummarizationModelError(
            "Failed to clean LLM response"
        ) from e
    
    except Exception as e:
        raise SummarizationModelError(
            "LLM model inference failed"
        ) from e
    
    finally:
                
        if model is not None:
            del model
        
        if outputs is not None:
            del outputs
            
        if inputs is not None:
                    del inputs
                  
        if tokenizer is not None:  
            del tokenizer
            
        clear_gpu_cache()

    
