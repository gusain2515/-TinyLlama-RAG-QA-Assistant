# TinyLlama RAG (Retrieval-Augmented Generation) System

This repository contains two main components:

1. A **Streamlit-based RAG assistant** that uses TinyLlama for natural language queries with document retrieval.
2. An **evaluation script** that measures the model's performance using BLEU, ROUGE, and BERTScore against a dataset of expected answers.

---

## 📁 Directory Structure

```
├── chromadb_index/               # Persistent vector store for document embeddings
├── Docs/                         # Folder with .pdf or .txt documents
├── rag_evaluation_dataset.csv   # CSV with questions and reference answers for evaluation
├── app.py                        # Streamlit-based QA interface
├── evaluat.py                    # Evaluation script with metrics
```

---



## 🚀 Streamlit QA App (`code.py`)

This script launches a web interface using Streamlit, allowing users to:

* Input a query.
* Retrieve top-3 relevant document chunks using Chroma vector store.
* Generate a short answer using TinyLlama.
* Show latency metrics for retrieval and LLM generation.

### 🖥️ Run the app

```bash
streamlit run code.py
```

### 🔍 Features

* Uses `sentence-transformers/all-MiniLM-L6-v2` for embedding.
* Retrieves top-3 documents and removes duplicate content.
* Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as the language model.
* Displays:

  * Response
  * Snippets from retrieved docs
  * Latency metrics (retrieval, LLM, total time)

---

## 📊 Evaluation Script (`evaluate.py`)

This script evaluates the end-to-end RAG system using a CSV dataset with queries and reference answers.

### 📁 CSV Format

The CSV should contain:

```
query,expected_answer
"What is RAG?","Retrieval-Augmented Generation is..."
```

### 📈 Metrics Used

* **BLEU** – Evaluates word overlap.
* **ROUGE-1 & ROUGE-L** – Recall-based scoring of n-grams.
* **BERTScore (F1)** – Semantic similarity using pre-trained embeddings.

### ▶️ Run Evaluation

```bash
python evaluate.py
```

### 📤 Output

Example output:

```
 BLEU Score:        0.0108
 ROUGE-1 Score:     0.0706
 ROUGE-L Score:     0.0617
 BERTScore (F1):    0.8097
```

