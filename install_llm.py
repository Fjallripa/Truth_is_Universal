# Locally install certain HuggingFace LLMs
# ========================================
# Run `python install_llm.py -h` for usage instructions.

# Co-written with Claude Sonnet 3.7


import argparse
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and save HuggingFace LLMs locally with specified options.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "-m", "--model", 
        nargs="+", 
        choices=get_supported_models().keys(),
        help="One or more models to download (space separated)"
    )
    model_group.add_argument(
        "-a", "--all", 
        action="store_true", 
        help="Download all supported models"
    )
    
    return parser.parse_args()



def get_supported_models():
    """
    Return a dictionary of supported models with their repository and save paths.
    Makes it easy to add more models in the future.
    """
    return {
        "llama3-8b": {
            "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
            "save_dir": "./weights/llama3_hf/llama3_8b_chat_hf"
        },
        "gemma2-9b": {
            "repo": "google/gemma-2-9b-it",
            "save_dir": "./weights/gemma2_hf/gemma2_9b_hf"
        },
        "mistral3-7b": {
            "repo": "mistralai/Mistral-7B-Instruct-v0.3",
            "save_dir": "./weights/mistral_hf/mistral_7b_it_hf_v3"
        }
        # Add more models here in the future
    }



def download_and_save_model(model_name, model_info):
    """Download a model and its tokenizer and save them locally."""
    print(f"\n{'='*60}")
    print(f"Processing {model_name}:")
    print(f"Repository: {model_info['repo']}")
    print(f"Save directory: {model_info['save_dir']}")
    print(f"{'='*60}\n")
    
    # Download tokenizer and model
    print(f"Downloading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_info['repo'])
    
    print(f"Downloading {model_name} model...")
    model = AutoModelForCausalLM.from_pretrained(model_info['repo'])
    
    # Save tokenizer and model
    print(f"Saving {model_name} tokenizer...")
    tokenizer.save_pretrained(model_info['save_dir'])
    
    print(f"Saving {model_name} model...")
    model.save_pretrained(model_info['save_dir'])
    
    # Clean up memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{model_name} successfully downloaded and saved.\n")



def main():
    """Main function to download and save models."""
    args = parse_arguments()
    supported_models = get_supported_models()
    
    # Determine which models to download
    models_to_download = []
    if args.all:
        models_to_download = list(supported_models.keys())
        print(f"Preparing to download all {len(models_to_download)} supported models.")
    elif args.model: 
        models_to_download = args.model
        print(f"Preparing to download {len(models_to_download)} selected model(s).")
    else:
        print(f"You need to specify at least one model to install. Use -h for help.")
    
    # Download and save each model sequentially
    for model_name in models_to_download:
        download_and_save_model(model_name, supported_models[model_name])
    
    print(f"\nAll requested models ({len(models_to_download)}) have been downloaded and saved.")





if __name__ == "__main__":
    main()
