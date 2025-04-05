import streamlit as st
import faiss, pickle, openai
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Load FAISS index and chunks
index = faiss.read_index("vector.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
openai.api_key = "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"

def semantic_search(query, top_k=5):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

def get_answer(query):
    docs = semantic_search(query)
    context = "\n\n".join(docs)
    prompt = f"""Answer the following question using the context below.

Context:
{context}

Question: {query}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("📚 RAG Q&A with OpenAI + Local Embeddings")
query = st.text_input("Enter your question:")
if query:
    st.write("Answer:")
    st.write(get_answer(query))

