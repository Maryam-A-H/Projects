import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
data = pd.read_csv("Train.csv")
data = data.iloc[:, 1:3].reset_index(drop=True)

# Load embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Embed texts
print("Embedding started...")
embeddings = embedder.encode(data['TEXT'].tolist(), convert_to_tensor=False)
print("Embedding completed.")

# Convert to numpy array
embeddings_array = np.array(embeddings)

# Save embeddings
np.save("train_embeddings.npy", embeddings_array)

# Save sampled data with indices for future merging
data.to_csv("Train_processed.csv", index=False)

print("Embeddings and processed data saved.")
