"""
Source: https://docs.vllm.ai/en/stable/features/quantization/int8.html
"""
import quantize

def main():

    hf_username = "ThatsGroes"

    models = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]

    for model in models:

        quantize.main(model=model, hf_username=hf_username)


if __name__ == "__main__":
    main()