import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert mathematical problem solver. "
            "Think step-by-step inside <think>...</think>. "
            "Write executable Python code in <code>...</code> blocks if calculation is needed. "
            "Put your final numerical answer in <answer>...</answer> tags."
        )
    },
    {"role": "user", "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=256, temperature=0.7, do_sample=True)

gen_tokens = out[0][input_ids.shape[1]:]
completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)

print("=" * 60)
print("RAW COMPLETION:")
print(completion)
print("=" * 60)
