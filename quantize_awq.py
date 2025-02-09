"""
Source: https://docs.vllm.ai/en/stable/features/quantization/int8.html
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import argparse


def main(model, hf_username="ThatsGroes"):

    print(f"Will quantize {model}")

    ###
    # Loading the Model
    ###

    MODEL_ID = model

    model_path = model
    SAVE_DIR = hf_username + "/" + MODEL_ID.split("/")[1] + "-AWQ"
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    #model.save_quantized(quant_path)
    #tokenizer.save_pretrained(quant_path)

    #print(f'Model is quantized and saved at "{quant_path}"')
    
    try:
        model.push_to_hub(SAVE_DIR, save_compressed=True)
        tokenizer.push_to_hub(SAVE_DIR)

    except Exception as e:
        print(f"Enter exception due to\n: {e}")
        model.push_to_hub(SAVE_DIR)
        tokenizer.push_to_hub(SAVE_DIR)
        
    try:
        print(f"Will save locally to {SAVE_DIR}")
        model.save_quantized(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

    except Exception as e:
        print(f"Saving locally failed due to error:\n {e}")

if __name__ == "__main__":

    #parser = argparse.ArgumentParser(
    #    prog='ProgramName',
    #    description='What the program does',
    #    epilog='Text at the bottom of help')
    
    #parser.add_argument('model_id', type=str)           # positional argument

    #args = parser.parse_args()

    #model_id = args.model_id

    models = ["meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct"]
    
    for model_id in models:

        main(model=model_id)