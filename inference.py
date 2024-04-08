"""
Code adapted from https://github.com/sashavor/co2_inference/blob/main/code/summarize/summarize_cnn.py

"""
from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer

from codecarbon import EmissionsTracker
from huggingface_hub import HfApi, ModelFilter
import logging
import torch
import huggingface_hub
import einops
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

# Create a dedicated logger (log name can be the CodeCarbon project name for example)
_logger = logging.getLogger("summarize_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/summarize_testing_cnn_final.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

summarization_models = ["microsoft/phi-2"]

### Load prompting datasets
from datasets import load_dataset, Dataset

def dset_gen():
    dset = load_dataset("cnn_dailymail", "3.0.0", split= 'test', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        yield row

dataset = Dataset.from_generator(dset_gen)

base_prompt = "You're an investigative journalist. Please make a brief summary of the main points from this text: "


torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)


for sum_model in summarization_models:
    inputs = base_prompt + d["article"]
    print(sum_model)
    tracker = EmissionsTracker(project_name=sum_model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/summarize_cnn_15.csv')
    tracker.start()
    tracker.start_task("load model")
    summarize = pipeline("summarization", model=sum_model, device=0,  trust_remote_code=True)
    model_emissions = tracker.stop_task()
    tracker.start_task("query model")
    count=0
    for d in dset:
        count+=1
        summarize(d['article'], max_length= 15, min_length=10)
    print('================'+str(count)+'================')
    model_emissions = tracker.stop_task()
    _ = tracker.stop()
    torch.cuda.empty_cache()