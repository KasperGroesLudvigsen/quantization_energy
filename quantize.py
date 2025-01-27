"""
Source: https://docs.vllm.ai/en/stable/features/quantization/int8.html
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


hf_username = "ThatsGroes"

models = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]

for model in models:

    ###
    # Loading the Model
    ###

    MODEL_ID = model

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ###
    # Preparing Calibration Data
    ###
    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048

    # Load and preprocess the dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)


    # Configure the quantization algorithms
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save the compressed model
    SAVE_DIR = hf_username + "/" + MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"

    try:
        model.push_to_hub(SAVE_DIR, save_compressed=True)
        tokenizer.push_to_hub(SAVE_DIR)

    except Exception as e:
        print(f"Enter exception due to\n: {e}")
        model.push_to_hub(SAVE_DIR)
        tokenizer.push_to_hub(SAVE_DIR)
