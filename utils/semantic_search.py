import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_similar_embeddings(df, query_embedding, top_k=5):
    embeddings = np.vstack(df['embedding'].values)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    df['similarity'] = similarities
    return df.sort_values(by='similarity', ascending=False).head(top_k)
