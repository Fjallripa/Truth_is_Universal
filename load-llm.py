from transformers import AutoTokenizer, AutoModelForCausalLM

repository_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#repository_name = "google/gemma-2-9b-it"
#repository_name = "mistralai/Mistral-7B-Instruct-v0.3"

save_dir = "./weights/llama3_hf/llama3_8b_chat_hf"
#save_dir = "./weights/gemma2_hf/gemma2_9b_hf"
#save_dir = "./weights/mistral_hf/mistral_7b_it_hf_v3"

print(f"Loading {repository_name} model and tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(repository_name)
model = AutoModelForCausalLM.from_pretrained(repository_name)

print(f"Saving both to {save_dir}.")
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Done.")
