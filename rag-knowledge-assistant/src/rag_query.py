def build_prompt(query, context):
    return f"""
You are an AI assistant. Answer the question using ONLY the context below.
If answer not in context, say "I don't know."

Question: {query}

Context:
{context}

Answer:
"""
