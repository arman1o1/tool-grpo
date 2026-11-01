import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_id = "Qwen/Qwen3.5-2B"
print(f"[*] Checking config for {model_id}...")
try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"[+] Loaded config: {config.model_type}")
except Exception as e:
    print(f"[-] Config error: {e}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"[+] Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
except Exception as e:
    print(f"[-] Tokenizer error: {e}")
