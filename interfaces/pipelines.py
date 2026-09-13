import torch
from util.utils import clear_gpu_cache, clean_response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, pipeline

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
    return quantizer_config

def audio_pipeline(model, audio_file, pipeline_task) -> str:
    """"
    This function contains the pipeline to create a text transcription from audio file
    """
    print(f"Model path: {model}")
    pipe = pipeline(
        pipeline_task,
        model=model,
        dtype=torch.float16,
        device='cuda',
        return_timestamps=True
    )

    result = pipe(audio_file)
    transcription = result["text"]
    
    del pipe
    clear_gpu_cache()
    
    return transcription
    
def llm_pipeline(model, messages, max_new_output_tokens) -> str:
    """"
    This function takes in a transcription as text and returns a textual response
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"Autokenised done...")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokeniser pad token done...")
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                           add_generation_prompt=True, 
                                           return_dict=False,
                                           ).to("cuda")
    print(f"shape: {inputs.shape}")
    input_length = inputs.shape[1]
    # streamer = TextStreamer(tokenizer)
    print(f"Converted input...")
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quantizer())
    print(f"Model loaded...")
    outputs = model.generate(inputs, max_new_tokens=max_new_output_tokens)
    print(f"Output done...")
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    print(f"Response: {response}")
    
    del inputs
    del tokenizer
    del model
    del outputs
    clear_gpu_cache()
    
    response = clean_response(response=response)
    
    return response

    
