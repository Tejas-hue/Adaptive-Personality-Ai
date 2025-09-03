import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- Load Models ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/gemma-2b-it"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./") # Loads adapter from the same folder

def generate_response(personality_profile, user_prompt):
    prompt = f"You are a chatbot. Your personality is: {personality_profile}. A user says: '{user_prompt}' Respond as yourself."
    input_text = f"<s>[INST] {prompt} [/INST]"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_response = response.split("[/INST]")[-1].strip().replace("</s>", "").strip()
    return cleaned_response

gr.Interface(
    fn=generate_response, 
    inputs=["text", "text"], # Two text inputs: one for profile, one for user prompt
    outputs="text"
).launch()
