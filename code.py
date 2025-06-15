import streamlit as st
import os
import torch
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Configuration
# ----------------------------
DB_DIR = "chromadb_index"
DOC_FOLDER = r"C:\Users\Lenovo\OneDrive\Desktop\rkm\Immv\Docs"

# ----------------------------
# Load Embeddings
# ----------------------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# ----------------------------
# Load or Create Chroma Vector DB
# ----------------------------
@st.cache_resource
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

# ----------------------------
# Load TinyLlama LLM Pipeline
# ----------------------------
@st.cache_resource
def load_tinyllama_pipeline():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="TinyLlama RAG QA", layout="centered")
st.title("🦙 TinyLlama CPU RAG Assistant")

query = st.text_input("Ask your question:", placeholder="e.g., What is retrieval-augmented generation?")

if query:
    with st.spinner("Retrieving documents and generating response..."):
        total_start = time.time()

        # Load database and retrieve top-k documents
        db = load_vector_store()
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        # Remove duplicate documents
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        context = "\n\n".join([doc.page_content for doc in unique_docs])

        # Load LLM and build prompt
        pipe = load_tinyllama_pipeline()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on the following context, give a very short answer: {query}\n\nContext:\n{context}"}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Run inference
        llm_start = time.time()
        outputs = pipe(prompt, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.95)
        llm_time = time.time() - llm_start
        total_time = time.time() - total_start

        # Clean up output
        response_text = outputs[0]["generated_text"]
        if "<|assistant|>" in response_text:
            response_text = response_text.split("<|assistant|>")[-1].strip()

        # ----------------------------
        # Display Results
        # ----------------------------
        st.markdown("### 📄 Response:")
        st.write(response_text)

        st.markdown("### 📚 Retrieved Context:")
        for i, doc in enumerate(unique_docs, 1):
            snippet = doc.page_content.strip().replace("\n", " ")
            st.markdown(f"**Doc {i}:** {snippet[:300]}{'...' if len(snippet) > 300 else ''}")

        st.markdown("### ⏱️ Latency Breakdown:")
        st.markdown(f"- **Retrieval Time:** {total_time - llm_time:.2f} seconds")
        st.markdown(f"- **LLM Inference Time:** {llm_time:.2f} seconds")
        st.markdown(f"- **Total Time:** {total_time:.2f} seconds")
