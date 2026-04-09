from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline

# Read system prompt from file
with open("./prompts/system/systemprompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Read user prompt from maria.json
with open("./prompts/user/maria.txt", "r", encoding="utf-8") as f:
    user_prompt = f.read()

def messages_to_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"<s>[INST] {msg['content']} "
        elif msg["role"] == "user":
            prompt += f"{msg['content']} [/INST]\n"
        elif msg["role"] == "assistant":
            prompt += f"{msg['content']}\n"
    return prompt

model_path = "../models/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

while True:
    user_input = input("Type 'exit' to quit: ")
    if user_input.strip().lower() == "exit":
        break
    prompt = messages_to_prompt(messages)
    output = generator(prompt, max_new_tokens=64, do_sample=False)[0]['generated_text']
    reply = output[len(prompt):].strip()
    print("Assistant:", reply)
    messages.append({"role": "assistant", "content": reply})