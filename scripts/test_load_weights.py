import torch
from transformers import AutoModelForCausalLM, AutoModel

model_id = "Qwen/Qwen3.5-2B"
print(f"[*] Testing AutoModelForCausalLM on {model_id}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    print(f"[+] Successfully loaded model: {type(model).__name__}")
    print(f"[+] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"[-] AutoModelForCausalLM failed: {e}")
    try:
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
        print(f"[+] Successfully loaded with AutoModel: {type(model).__name__}")
    except Exception as e2:
        print(f"[-] AutoModel also failed: {e2}")
