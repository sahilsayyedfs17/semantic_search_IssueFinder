from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
import torch

# Check if GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the CSV file
df = pd.read_csv("cleaned_VTVAS.csv")
print(f"Total records in CSV: {len(df)}")  # 192368

# Ensure Description and Summary columns are treated as strings and handle NaN values
df['Description'] = df['Description'].fillna('').astype(str)
df['Summary'] = df['Summary'].fillna('').astype(str)

# Concatenate Description and Summary fields
corpus = (df["Description"] + ". " + df["Summary"]).tolist()

# Function to encode in batches with progress bar
def encode_with_progress(corpus, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="Encoding sentences"):
        batch = corpus[i:i+batch_size]
        batch_embeddings = embedder.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Load the SBERT model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder = embedder.to(device)

# Create embeddings from concatenated fields
print("Creating embeddings...")
corpus_embeddings = encode_with_progress(corpus)

#Expiry badge and text label not displayed in PI for Catch-up content with multiple providers
# Query input
query = 'expiry icon and heading not visible for Catchup events'
top_k = 30

# Find the closest top_k sentences in the corpus based on cosine similarity
print("Finding similar sentences...")
query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
hits = hits[0]  # Get the hits for the first query

# Print the most similar sentences
print(f"\nTop {top_k} most similar sentences in corpus:")
for hit in hits:
    hit_id = hit['corpus_id']
    article_data = df.iloc[hit_id]
    title = article_data["Summary"]
    print("-", title, "(Score: {:.4f})".format(hit['score']))
