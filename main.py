"""
Measure energy consumption from the following models:
Llama 8b
Llama 8b Q8
Llama 8b Q4

Llama 3b
Llama 3b Q8
Llama 3b Q4

"""

from datasets import load_dataset, Dataset
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codecarbon import EmissionsTracker
import argparse
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from copy import copy
import torch 
import os
#from get_topics import get_topics
import random 
from dotenv import load_dotenv
from get_topics import get_topics


topics = get_topics()

groups = ["friends", "colleagues", "scientists", "politicians", "carpenters", "nurses", "doctors", "journalists", "tourists", "engineers"]

# TODO: Add to prompt that conversation must be realistic with stopped sentences etc. 
def make_prompt() -> dict:

    topic = random.choice(topics)

    group = random.choice(groups)

    prompt_options = [
        f"A group of {group} is having a conversation. What do you imagine they talk about? Write a text of 350-400 words that could pass as a conversation in this group. Be creative. Do not indicate speaker turns. Do not use quotation marks. Just write the conversation as one long text. Then, under the headline '**Summary**' write one sentence that summarizes the conversation, emphasizing any meetings, persons or places mentioned in the conversation. When you write the summary, imagine it is a transcription of an audio recording and that you do not know how many speakers are in the audio and you do not know what group they are.",
        f"Please write a text of 350-400 words that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns. Do not use quotation marks. Just write the transcription as one long text. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation. Do not indicate in the summary how many people were participating in the conversation.", # Indicate speaker turns like this: '**Speaker1**', '**Speaker2**' and so forth.
        f"Imagine you walked into a room where two or more people were in the middle of having a conversation on the topic of: {topic}. Write a verbatim transcript of 350-400 words of what they said. Do not indicate speaker turns. Do not use quotation marks. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation. Do not indicate in the summary how many people were participating in the conversation."
    ]

    #prompt = f"Please write a text that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns and do not use quotation marks. Just write the transcription as on long text. Then, write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation"
    prompt = random.choice(prompt_options)

    return {"prompt": [{"role": "user", "content": prompt}]}


# samples per iteration
SAMPLES = 10000

model_config = [{"model" : "meta-llama/Llama-3.1-8B-Instruct" ,"quantization": "bitsandbytes", "load_format" :"bitsandbytes"},
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                {"model" : "meta-llama/Llama-3.2-3B-Instruct" ,"quantization": "bitsandbytes", "load_format" :"bitsandbytes"}]


top_p = 0.95

temperature = 0.9

load_dotenv()

token = os.getenv("HF_TOKEN")

login(token, add_to_git_credential=True)

prompts = [make_prompt() for i in range(SAMPLES)]

dataset = Dataset.from_list(prompts)

df = dataset.to_pandas()
df.to_csv("dataset.csv", index=False)

token_df = []

for model in model_config:

    print(f"Starting inference with model: {model}")

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=2048*2)

    if isinstance(model, dict):

        llm = LLM(model=model["model"], max_seq_len_to_capture=8000, quantization=model["quantization"], load_format=model["load_format"])
        model = model["model"]
        project_name = model + "_quant"

    else:
        llm = LLM(model=model, max_seq_len_to_capture=8000)
        project_name=model

    token = os.getenv("HF_TOKEN") 

    tokenizer = AutoTokenizer.from_pretrained(model, token=token)
    

    # Log some GPU stats before we start inference
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(
        f"You're using the {gpu_stats.name} GPU, which has {max_memory:.2f} GB of memory "
        f"in total, of which {start_gpu_memory:.2f}GB has been reserved already."
    )

    tracker = EmissionsTracker(project_name=project_name, measure_power_secs=1)

    print("Starting inference..")

    tracker.start()

    outputs = llm.chat(dataset["prompt"], sampling_params)

    emissions = tracker.stop()

    responses = [output.outputs[0].text for output in outputs]

    tokens = [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]

    total_tokens = sum(tokens)

    print(f"Total tokens generated by {model}:\n {total_tokens}")

    token_df.append({"model" : model, "num_tokens": total_tokens})

    torch.cuda.empty_cache()

    # torch.cuda.empty_cache does not properly free up memory
    del llm 
    del tokenizer

    print(f"\nEmissions from generating queries with {model}:\n {emissions}")
    energy_consumption_kwh = tracker._total_energy.kWh  # Total energy in kWh
    print(f"\nEnergy consumption from generating queries with {model}:\n {energy_consumption_kwh}")

    # Print some post inference GPU stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_inference = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    inference_percentage = round(used_memory_inference / max_memory * 100, 3)

    print(
        f"We ended up using {used_memory:.2f} GB GPU memory ({used_percentage:.2f}%), "
        f"of which {used_memory_inference:.2f} GB ({inference_percentage:.2f}%) "
        "was used for inference."
    )

token_df = pd.DataFrame(token_df)
token_df.to_csv("total_tokens.csv", index=False)