"""
Source: https://docs.vllm.ai/en/stable/features/quantization/int8.html
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier



def main(model, hf_username="ThatsGroes"):

    ###
    # Loading the Model
    ###

    MODEL_ID = model

    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Configure the simple PTQ quantization
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

    # Apply the quantization algorithm.
    oneshot(model=model, recipe=recipe)

    # Save the compressed model
    SAVE_DIR = hf_username + "/" + MODEL_ID.split("/")[1] + "-FP8-Dynamic"

    try:
        model.push_to_hub(SAVE_DIR, save_compressed=True)
        tokenizer.push_to_hub(SAVE_DIR)

    except Exception as e:
        print(f"Enter exception due to\n: {e}")
        model.push_to_hub(SAVE_DIR)
        tokenizer.push_to_hub(SAVE_DIR)
