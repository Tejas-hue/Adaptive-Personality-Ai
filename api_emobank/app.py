import gradio as gr
import torch
import joblib
from transformers import AutoTokenizer, RobertaPreTrainedModel, RobertaModel
from torch.nn import MSELoss

class RobertaForRegression(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        cls_token_output = outputs[0][:, 0, :]
        logits = self.classifier(cls_token_output)
        preds = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(preds.squeeze(), labels.squeeze())
        return {"loss": loss, "logits": preds}

print("Loading EmoBank model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./") # Loads from the same folder
model = RobertaForRegression.from_pretrained("./").to(device)
scaler = joblib.load("scaler_emobank.pkl")
labels = ['Valence (Positivity)', 'Arousal (Energy)', 'Dominance (Control)']
print("EmoBank model loaded successfully.")

def predict_emobank_scores(text):
    """Predicts VAD scores for a given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        preds_scaled = model(**inputs)['logits']
    preds_original = scaler.inverse_transform(preds_scaled.cpu().numpy())[0]
    results = {label: float(score) for label, score in zip(labels, preds_original)}
    return results

gr.Interface(
    fn=predict_emobank_scores, 
    inputs=gr.Textbox(lines=5, label="Input Text"), 
    outputs="json",
    title="EmoBank VAD Meter",
    description="An API for the fine-tuned RoBERTa-large model that predicts Valence, Arousal, and Dominance from text."
).launch()
