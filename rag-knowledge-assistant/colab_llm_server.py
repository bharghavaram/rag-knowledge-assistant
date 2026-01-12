from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate(req: Prompt):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"answer": ans}
