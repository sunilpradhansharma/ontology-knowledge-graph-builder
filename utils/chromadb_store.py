import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def create_chromadb_collection(df, collection_name="rdf_attributes"):
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)

    for idx, row in df.iterrows():
        collection.add(
            ids=[row['entity_uri']],
            documents=[row['label']],
            metadatas=[{"label": row['label']}]
        )
    return collection
