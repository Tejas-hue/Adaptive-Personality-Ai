# Personality & Emotion Analysis from Text

## Overview

This project focuses on predicting personality traits and emotions from text. The workflow progresses from simple baseline models to more complex deep learning approaches, allowing direct comparison of performance across different tasks and datasets.  

The aim is to explore the effectiveness of traditional machine learning versus transformer-based models for text-based personality and emotion analysis.

---

## Datasets

Four datasets were used in this project:

* **Essaysbig5** – Essays with binary labels for the Big Five personality traits.
* **GoEmotions** – Reddit comments annotated with 27 possible emotion labels.
* **Pandora** – Text with continuous scores (0-100) for personality traits.
* **EmoBank** – Text with continuous Valence, Arousal, and Dominance (VAD) scores.

You can replace or extend these datasets in the future to test new models or tasks.

---

## Methodology

Three modeling approaches were applied:

1. **Baselines**  
   Simple ML models (Naive Bayes, Logistic Regression) using **TF-IDF** features to establish a performance baseline.

2. **Sentence Embeddings**  
   Pre-trained sentence embeddings (`all-MiniLM-L6-v2`) were used as features for the same ML models to see improvements from contextual embeddings.

3. **Transformer Fine-Tuning**  
   Fine-tuned a `DistilBERT` transformer model for each dataset to leverage deep contextual understanding of text.

---

## Challenges & Solutions

### Transformer Regression Instability
* **Problem:** Regression tasks on Pandora and EmoBank resulted in unstable training and negative R² scores.
* **Solution:** Normalized target labels to 0-1 and added a Sigmoid activation to constrain predictions.

### Environment & Dependency Issues
* **Problem:** Outdated `transformers` library in Colab caused repeated `TypeError`.
* **Solution:** Simplified `TrainingArguments` to a backward-compatible minimal set.

### Resource Management
* **Problem:** Large datasets filled Colab’s temporary storage.
* **Solution:** Model checkpoints were saved directly to Google Drive instead of the local disk.

---

## Performance Comparison

| Dataset | Task | Metric | Baseline | Fine-Tuned Transformer | Best Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Essaysbig5 | Classification | F1-Score | ~0.65 | 0.6023 | Baseline (Naive Bayes) |
| GoEmotions | Classification | F1-Score | ~0.23 | 0.3416 | Fine-Tuned Transformer |
| Pandora | Regression | R² Score | ~0.06 | 0.1158 | Fine-Tuned Transformer |
| EmoBank | Regression | R² Score | ~0.15 | 0.3755 | Fine-Tuned Transformer |

This table highlights clear improvements in most tasks from transformer fine-tuning.

---

## Models

All final models are stored in the `v1_models/` directory:

* `best_model_essaysbig5_naivebayes.pkl` – Best for Essaysbig5 classification.
* Transformer models for GoEmotions, Pandora, and EmoBank tasks are also included.

---

## Next Steps

Potential extensions and improvements:

* Experiment with larger transformers like `BERT-base` or `RoBERTa`.
* Systematic hyperparameter tuning for all models.
* Convert low-performing regression tasks into classification by discretizing continuous labels.
* Integrate models into a conversational AI system.
* Add automated evaluation scripts for new datasets.
* Build visualizations for model predictions and feature importance.





