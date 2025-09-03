import gradio as gr
import joblib

# Load model and vectorizer from the same folder
model = joblib.load("best_model_essaysbig5_naivebayes.pkl")
vectorizer = joblib.load("vectorizer_essaysbig5_tfidf.pkl")
labels = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

def predict_personality(text):
    input_tfidf = vectorizer.transform([text])
    preds = model.predict(input_tfidf)[0]
    results = {label: ("High" if pred == 1 else "Low") for label, pred in zip(labels, preds)}
    return results

gr.Interface(fn=predict_personality, inputs="text", outputs="json").launch()
