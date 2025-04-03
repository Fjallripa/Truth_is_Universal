from transformers import AutoTokenizer, AutoModelForCausalLM

repository_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#repository_name = "roneneldan/tinystories-1M"
print(f"Loading {repository_name} model and tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(repository_name)
model = AutoModelForCausalLM.from_pretrained(repository_name)

save_dir = "./weights/llama3_hf/llama3_8b_chat_hf"
#save_dir = "./weights/tinystories_hf/tinystories_1m_hf"
print(f"Saving both to {save_dir}.")
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Done.")
