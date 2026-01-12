import streamlit as st
import requests
from load_pdf import load_pdf
from embed import chunk_text, get_embeddings_model, embed_chunks
from vector_store import create_vector_store, add_to_store, query_store
from rag_query import build_prompt

LLM_URL = "PASTE_YOUR_COLAB_URL_HERE/generate"

st.title("📚 RAG Knowledge Assistant (Chat Mode)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask your question about the document")

if query:
    st.session_state.history.append(("user", query))
    text = load_pdf("data/sample.pdf")
    chunks = chunk_text(text)
    model = get_embeddings_model()
    embeddings = embed_chunks(model, chunks)
    collection = create_vector_store()
    add_to_store(collection, chunks, embeddings)
    q_embed = model.encode([query])[0]
    results = query_store(collection, q_embed)
    top_chunk = results["documents"][0][0]
    prompt = build_prompt(query, top_chunk)
    resp = requests.post(LLM_URL, json={"prompt": prompt})
    answer = resp.json()["answer"]
    st.session_state.history.append(("assistant", answer))

for role, content in st.session_state.history:
    with st.chat_message("assistant" if role=="assistant" else "user"):
        st.write(content)
