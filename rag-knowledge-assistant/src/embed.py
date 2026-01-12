from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(model, chunks):
    return model.encode(chunks)
