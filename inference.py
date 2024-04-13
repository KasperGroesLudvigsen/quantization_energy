"""
Code adapted from https://github.com/sashavor/co2_inference/blob/main/code/summarize/summarize_cnn.py

"""
from datasets import load_dataset, Dataset
from codecarbon import EmissionsTracker
from huggingface_hub import HfApi, ModelFilter
import logging
import torch
#import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

import os
from llama_cpp import Llama


llm = Llama(
model_path="models--TheBloke--phi-2-GGUF/snapshots/5a454d977c6438bb9fb2df233c8ca70f21c87420/phi-2.Q4_K_M.gguf",  # Download the model file first
n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
)

def inference(prompt):

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.


    # Simple inference example
    output = llm(
    "Instruct: {prompt}\nOutput:", # Prompt
    max_tokens=512,  # Generate up to 512 tokens
    stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=True        # Whether to echo the prompt
    )

    return output

    # Chat Completion API

    #llm = Llama(model_path="./phi-2.Q4_K_M.gguf", chat_format="llama-2")  # Set chat_format according to the model you are using
    #llm.create_chat_completion(
    #    messages = [
    #        {"role": "system", "content": "You are a story writing assistant."},
    #        {
    #            "role": "user",
    #            "content": "Write a story about llamas."
    #        }
    #    ]
    #)


# Create a dedicated logger (log name can be the CodeCarbon project name for example)
_logger = logging.getLogger("quantize_energy")
_channel = logging.FileHandler('quantize_energy.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

summarization_models = ["microsoft/phi-2"]

### Load prompting datasets

def dset_gen():
    dset = load_dataset("cnn_dailymail", "3.0.0", split= 'test', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        yield row

dataset = Dataset.from_generator(dset_gen)

base_prompt = "You're an investigative journalist. Please make a brief summary of the main points from this text:"


torch.set_default_device("cuda")

torch.cuda.is_available()

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

model_name = "TheBloke/phi-2-GGUF"
model_file = "phi-2.Q4_K_M.gguf" # this is the specific model file we'll use in this example. It's a 4-bit quant, but other levels of quantization are available in the model repo if preferred
model_path = hf_hub_download(model_name, filename=model_file)

quantized_model = AutoModelForCausalLM.from_pretrained("models--TheBloke--phi-2-GGUF", torch_dtype="auto", trust_remote_code=True)

#inputs = tokenizer(f"{base_prompt} {dset[0]["article"]}", return_tensors="pt", return_attention_mask=False)

#outputs = model.generate(**inputs, max_length=200)
#text = tokenizer.batch_decode(outputs)[0]
#print(text)

dset = Dataset.from_generator(dset_gen)

# Just to see a single input - the output makes no sense at all, but I guess it's just because the LLM is rubbish
inputs = tokenizer(f"{base_prompt} {dset[0]['article']}", return_tensors="pt", return_attention_mask=False)
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]



for sum_model in summarization_models:
    #inputs = base_prompt + d["article"]

    print(sum_model)
    tracker = EmissionsTracker(project_name=sum_model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/summarize_cnn_15.csv')
    tracker.start()
    tracker.start_task("load model")
    #summarize = pipeline("summarization", model=sum_model, device=0,  trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    model_emissions = tracker.stop_task()
    tracker.start_task("query model")
    #count=0
    for d in dset:
        inputs = tokenizer(f"{base_prompt} {d["article"]}", return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=200)
        text = tokenizer.batch_decode(outputs)[0]
        #count+=1
        #summarize(d['article'], max_length= 15, min_length=10)
    #print('================'+str(count)+'================')
    model_emissions = tracker.stop_task()
    _ = tracker.stop()
    torch.cuda.empty_cache()