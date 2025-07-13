import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
data = pd.read_csv("Train.csv")
data = data.iloc[:, 1:3].reset_index(drop=True)
data

# Embed
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(data['Text'].tolist(), convert_to_tensor=False)

# Save
np.save("train_embeddings.npy", embeddings)
print("Embeddings saved successfully.")
