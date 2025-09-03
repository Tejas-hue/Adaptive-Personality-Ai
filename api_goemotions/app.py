import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForSequenceClassification.from_pretrained("./").to(device)
labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.sigmoid(logits).squeeze().cpu().numpy()
    results = {label: float(score) for label, score in zip(labels, scores) if score > 0.1}
    return results

gr.Interface(fn=predict_emotions, inputs="text", outputs="json").launch()
