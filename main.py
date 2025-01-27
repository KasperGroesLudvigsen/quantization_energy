"""
Main script for generating synthetic dialogs
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

# samples per iteration
SAMPLES = 5000

models = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"]

iterations = 10

top_p = 0.95

temperature = 0.9

load_dotenv()

token = os.getenv("HF_TOKEN")

login(token, add_to_git_credential=True)

#models = ["google/gemma-2-27b-it"]

sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=2048*2)

llm = LLM(model=model, max_seq_len_to_capture=8000, )

#tokenizer = AutoTokenizer.from_pretrained(model, token=token)

topics = get_topics()

lengths = ["100-150 words", "250-300 words", "350-400 words", "600-700 words"]

groups = pd.read_csv("groups.csv")["group"].to_list()

# TODO: Add to prompt that conversation must be realistic with stopped sentences etc. 
def make_prompt() -> dict:

    topic = random.choice(topics)

    length = random.choice(lengths)

    group = random.choice(groups)

    prompt_options = [
        f"A group of {group} is having a conversation. What do you imagine they talk about? Write a text of {length} that could pass as a conversation in this group. Be creative. Do not indicate speaker turns. Do not use quotation marks. Just write the conversation as one long text. Then, under the headline '**Summary**' write one sentence that summarizes the conversation, emphasizing any meetings, persons or places mentioned in the conversation. When you write the summary, imagine it is a transcription of an audio recording and that you do not know how many speakers are in the audio and you do not know what group they are.",
        #f"A group of {group} is having a conversation. What do you imagine the topic of the conversation is? Be creative. You are not allowed to suggest topics about AI and ethics. You must output only the topic of the conversation.",
        #"Suggest a conversation topic. Your output should only be the topic - nothing else. Be creative. Here are some examples of topics:\n- Shopping\n-Christmas plan\n-arabic traditions\n-smuggling"
        f"Please write a text of {length} that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns. Do not use quotation marks. Just write the transcription as one long text. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation. Do not indicate in the summary how many people were participating in the conversation.", # Indicate speaker turns like this: '**Speaker1**', '**Speaker2**' and so forth.
        #f"Please write a text of {length} that could pass as a transcription of an everyday conversation between two or more people on a topic of your own choice. Be creative. Do not indicate speaker turns. Do not use quotation marks. Just write the transcription as one long text. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation.", # Indicate speaker turns like this: '**Speaker1**', '**Speaker2**' and so forth.
        #f"Please write a text of {length} that could pass as a transcription of a telephone conversation between a customer and a customer service representative on the topic of: {cs_topic}. Do not indicate speaker turns. Do not use quotation marks. Just write the transcription as one long text. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation.",
        f"Imagine you walked into a room where two or more people were in the middle of having a conversation on the topic of: {topic}. Write a verbatim transcript of {length} of what they said. Do not indicate speaker turns. Do not use quotation marks. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation. Do not indicate in the summary how many people were participating in the conversation."
    ]

    #prompt = f"Please write a text that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns and do not use quotation marks. Just write the transcription as on long text. Then, write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation"
    prompt = random.choice(prompt_options)

    return {"prompt": [{"role": "user", "content": prompt}]}

token = os.getenv("HF_TOKEN") 

all_results = []

energy_use = []

# Log some GPU stats before we start inference
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(
    f"You're using the {gpu_stats.name} GPU, which has {max_memory:.2f} GB of memory "
    f"in total, of which {start_gpu_memory:.2f}GB has been reserved already."
)

tracker = EmissionsTracker()
tracker.start()

for i in range(iterations):

    print(f"Starting iteration {i+1} / {iterations}")

    print(f"Starting to create {SAMPLES} prompts..")
    prompts = [make_prompt() for i in range(SAMPLES)]
    print("Finished creating prompts")

    dataset = Dataset.from_list(prompts)

    results = copy(dataset)

    print("Starting inference..")
    outputs = llm.chat(dataset["prompt"], sampling_params)
    
    responses = [output.outputs[0].text for output in outputs]

    results = results.add_column("response", responses)

    #results = results.add_column("model", [model for _ in range(len(results))])
    
    # number of tokens in the prompt and response. Used for calculcating kwh/token
    #results = results.add_column("num_tokens_query", [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]) # [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]

    # each element in results["prompt"] is a list with a dictionary with two keys: "content" and "role"
    #results = results.add_column("num_tokens_prompt", [len(tokenizer.encode(text[0]["content"], add_special_tokens=False)) for text in results["prompt"]])

    results.to_csv(f"summaries-iter-{i+1}-of-{iterations}.csv")

    all_results.append(results)

    #torch.cuda.empty_cache()

    # torch.cuda.empty_cache does not properly free up memory
    #del llm 

emissions = tracker.stop()
print(f"Emissions from generating queries with {model}:\n {emissions}")
energy_consumption_kwh = tracker._total_energy.kWh  # Total energy in kWh
print(f"Energy consumption from generating queries with {model}:\n {emissions}")

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

final_dataset = concatenate_datasets(all_results)

final_dataset.to_csv("summaries_all.csv")

final_dataset.push_to_hub("ThatsGroes/synthetic-dialog-summaries-raw")






# time it took to generate: print(outputs[0].metrics.finished_time - outputs[0].metrics.arrival_time)