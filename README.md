# Adaptive Personality & Emotion AI

---

## 1. Project Overview
This project is a **complete, end-to-end NLP pipeline** that trains, evaluates, and deploys a suite of AI models to predict human **personality** and **emotion** from text.  

I conducted a comparative study of multiple machine learning approaches, ranging from simple baselines to **large-scale transformers**, across four datasets.  
The final solution is deployed as a **suite of AI microservices** with a **central user interface**.

---

## 2. System Architecture

The system is implemented in a **microservices architecture**:

- **Five independent models** are deployed as APIs on Hugging Face Spaces (via Gradio).  
  - One of these is the generative **Gemma "Personality Brain"**.  
  - The other four are specialized "analyzer" or "skill" models.  

- **A sixth orchestrator app** (Streamlit dashboard) integrates everything.  
  - It calls the APIs in the background.  
  - Provides an interactive experience without hosting heavy models directly.  


---

## 3. Datasets & Tasks

| Dataset     | Task                                        | Type                        |
|-------------|---------------------------------------------|-----------------------------|
| EssaysBig5  | Personality prediction from essays          | Multi-output Classification |
| GoEmotions  | Emotion detection from Reddit comments      | Multi-label Classification  |
| Pandora     | Personality score prediction (short texts)  | Multi-output Regression     |
| EmoBank     | VAD (Valence, Arousal, Dominance) scores    | Multi-output Regression     |

---

## 4. Methodology & Models Tested

**Tiered benchmarking approach:**

- **Tier 1 — Baselines**  
  Naive Bayes, Linear SVM, Ridge Regression on TF-IDF features  

- **Tier 2 — Transformer (RoBERTa-base)**  
  Fine-tuned `roberta-base`  

- **Tier 3 — Transformer (RoBERTa-large)**  
  Fine-tuned `roberta-large` for SOTA performance  

- **Generative Model**  
  Fine-tuned `google/gemma-2b-it` → conversational **Personality Brain**

---

## 5. Results

**Main performance comparison (best baseline vs. RoBERTa-large):**

| Dataset     | Task            | Metric    | Best Baseline Score      | RoBERTa-large Score | Overall Winner     |
|-------------|-----------------|-----------|--------------------------|---------------------|--------------------|
| EssaysBig5  | Classification  | F1-Score  | 0.65 (Naive Bayes)       | 0.60                | Baseline           |
| GoEmotions  | Classification  | F1-Score  | 0.23 (Linear SVM)        | 0.36                | RoBERTa-large      |
| Pandora     | Regression      | R² Score  | 0.06 (Ridge)             | 0.15                | RoBERTa-large      |
| EmoBank     | Regression      | R² Score  | 0.13 (Ridge)             | 0.47                | RoBERTa-large      |

---

## 6. Key Technical Challenges Solved

- **Transformer Regression Instability**  
  - Diagnosed catastrophic negative R² scores  
  - Engineered custom `RobertaForRegression` model with sigmoid outputs  
  - Normalized targets → stable, positive results  

- **Resource Management under Constraints**  
  - Handled GPU timeouts, disk space errors, RAM crashes  
  - Solutions: checkpoint-and-resume, Google Drive storage, memory-efficient loading, quantization  

- **Dependency Conflicts & Data Corruption**  
  - Fixed version mismatches in Colab  
  - Forced model downloads in multiple formats to resolve corruption  

- **Deployment Pipeline**  
  - Solved Hugging Face Spaces issues: Git LFS errors, config bugs, API timeouts  
  - Built robust multi-part deployment strategy  

---

## 7. Live Demos

Deployed as interactive demos on Hugging Face Spaces:

- [Main Dashboard (Streamlit)](https://huggingface.co/spaces/Antta)  
- [Live Demo: Personality Brain (Gemma)](https://huggingface.co/spaces/Antta)  
- [Live Demo: GoEmotions Emotion Detector](https://huggingface.co/spaces/Antta)  
- [Live Demo: Pandora Personality Assessor](https://huggingface.co/spaces/Antta)  
- [Live Demo: EmoBank VAD Regressor](https://huggingface.co/spaces/Antta)  
- [Live Demo: EssaysBig5 Personality Classifier](https://huggingface.co/spaces/Antta)  

---

## 8. Future Work

- Conduct systematic **hyperparameter search** for each model  
- Explore larger models (e.g., **Llama 3, Mixtral**) in stronger cloud environments  
- Build an integrated **chatbot** where Gemma uses real-time outputs from analyzers  

---
## 9. Installation & Usage

Clone the repository:
```bash
git clone https://github.com/Tejas-hue/Adaptive-Personality-Ai.git
cd adaptive-personality-emotion-ai
```
Install dependencies:
```bash
pip install -r requirements.txt
```
