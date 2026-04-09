import torch
import json
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("../models/SmolVLM2-500M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "../models/SmolVLM2-500M-Video-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Read system prompt from file
with open("./prompts/system/systemprompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Read user prompt from maria.json
with open("./prompts/user/maria.txt", "r", encoding="utf-8") as f:
    user_prompt = f.read()

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"


conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt}
        ]
    },
    {
        "role": "user",
        "content":[
            # {"type": "image", "url": image_url},
            {"type": "text", "text": user_prompt}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

output_ids = model.generate(**inputs, max_new_tokens=64)
generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
print(generated_texts)