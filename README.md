# PsyCoMark – Conspiracy Detection & Psycholinguistic Marker Extraction  
### Group Project  
**Members:**  
- Abhinav Mehrotra  
- Benarjee Sudeep Sampath Pyla  
- Ajay Tata  

---

## 🔍 Project Overview

This repository contains our work for the **SemEval-2026 Task 10 – PsyCoMark**, which focuses on detecting conspiratorial thinking in Reddit posts through:

### **Subtask 1 — Psycholinguistic Marker Extraction**
Extract spans belonging to five markers:
- Actor  
- Action  
- Effect  
- Victim  
- Evidence  

### **Subtask 2 — Conspiracy Detection**
Classify each Reddit submission statement as:
- **Yes** (conspiracy-related)  
- **No** (not conspiracy-related)  
- **Can't tell** (ambiguous)

---

## 🚀 Project Progress Summary

So far, we have completed all foundational and preprocessing steps required for the project:

### ✔ Environment & Repo Setup
- Created a GitHub repository  
- Set up a clean project structure  
- Created a Conda environment (`psycomark`)  
- Installed all dependencies (Transformers, Datasets, Torch, etc.)  
- Added the environment as a Jupyter kernel  

### ✔ Data Rehydration (Major Milestone Achieved)
- Integrated the SemEval starter pack into our project  
- Found and fixed bugs in the rehydration script  
- Successfully rehydrated all available Reddit comments  
- Produced a clean `train_full.jsonl` dataset  

### ✔ Dataset Preparation
We created two processed datasets:

1. `classification_train.csv` → For conspiracy detection  
2. `ner_train.jsonl` → Span-based dataset for marker extraction  

These datasets are now ready for model training.

### ✔ Modeling Setup Completed
- Created separate notebooks:
  - `classification_training.ipynb`
  - `ner_training.ipynb`
- Initialized RoBERTa-base classifier  
- Tokenized and preprocessed the full dataset  
- Prepared TrainingArguments and Trainer pipeline  

We are now ready to proceed with **classifier training and NER modeling**.

---

## 📁 Repository Structure

---

## ⚙️ Setup Instructions (Reproducible Workflow)

### 1. Clone Repo
```bash
git clone https://github.com/sudeepben/psycomark-project
cd psycomark-project


2. Create Conda Environment
conda create -n psycomark python=3.10 -y
conda activate psycomark


3. Install Dependencies
pip install transformers datasets accelerate torch markdown beautifulsoup4


4. Rehydrate Data
python src/rehydration/rehydrate_data.py --input data/raw/train_redacted.jsonl --output data/processed/train_full.jsonl


5. Train Models
Use the notebooks in the notebooks/ directory.