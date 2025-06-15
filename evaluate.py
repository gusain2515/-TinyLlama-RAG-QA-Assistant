import os
import time
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import evaluate

# ------------------------
# Config
# ------------------------
DB_DIR = "chromadb_index"
DOC_FOLDER = r"C:\Users\Lenovo\OneDrive\Desktop\rkm\Immv\Docs"
CSV_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\rkm\Immv\rag_evaluation_dataset.csv"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ------------------------
# Load Embedding Model
# ------------------------
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# ------------------------
# Load Vector Store
# ------------------------
def load_vector_store():
    embedding_model = load_embedding_model()
    if not os.path.exists(os.path.join(DB_DIR, "index")):
        documents = []
        for filename in os.listdir(DOC_FOLDER):
            file_path = os.path.join(DOC_FOLDER, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(file_path)
            elif filename.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
            else:
                continue

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)
            documents.extend(split_docs)

        vectordb = Chroma.from_documents(documents, embedding=embedding_model, persist_directory=DB_DIR)
        vectordb.persist()
        return vectordb
    else:
        return Chroma(persist_directory=DB_DIR, embedding_function=embedding_model.encode)

# ------------------------
# Load TinyLlama Model
# ------------------------
def load_tinyllama_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# ------------------------
# Evaluate the model
# ------------------------
def evaluate_model():
    df = pd.read_csv(CSV_PATH)
    queries = df["query"].tolist()
    references = df["expected_answer"].tolist()
    predictions = []

    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    pipe = load_tinyllama_pipeline()

    print("Running evaluation...")
    for i, query in enumerate(queries):
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on the following context,give very short answer : {query}\n\nContext:\n{context}"}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        result = pipe(prompt, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.95)
        generated = result[0]["generated_text"]
        predictions.append(generated)
        print(f"[{i+1}/{len(queries)}] ✅")

    # Metrics
    print("\nCalculating metrics...\n")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")

    print(f"✅ BLEU Score:        {bleu_score['bleu']:.4f}")
    print(f"✅ ROUGE-1 Score:    {rouge_score['rouge1']:.4f}")
    print(f"✅ ROUGE-L Score:    {rouge_score['rougeL']:.4f}")
    print(f"✅ BERTScore (F1):   {sum(bert_score['f1']) / len(bert_score['f1']):.4f}")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    evaluate_model()
