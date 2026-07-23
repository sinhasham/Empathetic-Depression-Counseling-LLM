# rag/retriever.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag.knowledge_base import documents

corpus = [doc["topic"] + " " + doc["text"] for doc in documents]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(corpus)

def retrieve(query, emotion, top_k=3):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, doc_vectors)[0]

    for i, doc in enumerate(documents):
        if doc["topic"] == emotion:
            scores[i] += 1.0

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"topic": documents[i]["topic"], "text": documents[i]["text"], "score": round(scores[i], 4)} for i in top_indices]