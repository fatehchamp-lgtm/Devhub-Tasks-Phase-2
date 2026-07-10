# DevHub Tasks Phase 2 - Advanced AI Automation

This repository contains the tasks completed during Phase 2 of the Artificial Intelligence and Machine Learning Internship at **DevelopersHub Corporation**.

---

## Task 1: News Topic Classifier Using BERT

### 🎯 Objective
The primary goal of this task is to fine-tune a state-of-the-art Transformer model (`bert-base-uncased`) to automatically classify news headlines into four distinct categories from the standard **AG News Dataset**:
1. World
2. Sports
3. Business
4. Sci/Tech

---

### 🛠️ Methodology & Approach
The deployment follows a complete full-stack machine learning pipeline divided into three major stages:

1. **Data Preprocessing & Tokenization:**
   * Utilized Hugging Face `datasets` to stream the official AG News dataset.
   * Implemented `AutoTokenizer` from the `transformers` library to convert raw text headlines into input IDs and attention masks with a controlled maximum length of 128 tokens, implementing strict padding and truncation.

2. **Model Training & Fine-Tuning (Backend):**
   * Loaded a pre-trained `BertForSequenceClassification` model setup with 4 output labels.
   * Configured `TrainingArguments` using Hugging Face's advanced ecosystem paired with `accelerate` and `PyTorch` for CPU-optimized training loops.
   * Fine-tuned the weights across 2 full epochs with a learning rate of $2 \times 10^{-5}$ and evaluated performance flags at regular checkpoints.

3. **Live Deployment (Frontend):**
   * Developed an interactive, lightweight web UI using **Streamlit** (`app.py`).
   * The app caches and loads the fine-tuned local weights (`saved_bert_model`) seamlessly, taking real-time user text inputs to showcase live predictions along with class-wise probability breakdown charts.

---

### 📊 Key Results & Observations
* **Training Dynamics:** The cross-entropy training loss significantly dropped from an initial `1.027` down to a precise `0.2505`, demonstrating excellent convergence.
* **Final Evaluation Results:**
  * **Evaluation Accuracy:** `0.8900` (89.0%)
  * **F1-Score:** `0.8906` (89.06%)
* **Observation:** BERT drastically outperforms traditional baseline models (like standard TF-IDF or classic statistical representations) by capturing bidirectional semantic contexts, accurately identifying structural differences in short, fast-paced news headlines.

---

## 🚀 How to Run the App Locally

1. **Activate Virtual Environment:**
   ```bash
   .venv\Scripts\activate