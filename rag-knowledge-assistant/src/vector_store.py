import chromadb

def create_vector_store():
    client = chromadb.Client()
    return client.create_collection(name="docs")


def add_to_store(collection, chunks, embeddings):
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i]]
        )


def query_store(collection, query_embedding, top_k=3):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
