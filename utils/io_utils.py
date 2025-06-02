import pandas as pd
import pickle

def save_embeddings(df: pd.DataFrame, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)

def load_embeddings(file_path: str) -> pd.DataFrame:
    with open(file_path, 'rb') as f:
        return pickle.load(f)
