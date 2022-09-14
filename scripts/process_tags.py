import pandas as pd
import gensim as gs

def create_tag_embeddings(filepath):
    data = pd.read_parquet(filepath)
    

