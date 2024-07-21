from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
import torch
import os

# Check if GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the CSV file
csv_file_path = "cleaned_VTVAS.csv"
embeddings_file_path = "corpus_embeddings.pt"

df = pd.read_csv(csv_file_path)
print(f"Total records in CSV: {len(df)}")  # 192368

# Ensure Description and Summary columns are treated as strings and handle NaN values
df['Description'] = df['Description'].fillna('').astype(str)
df['Summary'] = df['Summary'].fillna('').astype(str)

# Concatenate Description and Summary fields
corpus = (df["Description"] + ". " + df["Summary"]).tolist()

# Load the SBERT model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder = embedder.to(device)

def encode_with_progress(corpus, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="Encoding sentences"):
        batch = corpus[i:i+batch_size]
        batch_embeddings = embedder.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Check if embeddings already exist
if os.path.exists(embeddings_file_path):
    print("Loading existing embeddings...")
    corpus_embeddings = torch.load(embeddings_file_path)
else:
    print("Creating embeddings...")
    corpus_embeddings = encode_with_progress(corpus)
    print("Saving embeddings...")
    torch.save(corpus_embeddings, embeddings_file_path)

#It has been observed that after Standby Wakeup zapping banner does not appear.
# Query input
query = 'After turning On from Standby , banner not visible'
top_k = 50

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
    title = article_data["Issue Key"] + "->  " + article_data["Summary"]
    print("-", title, "(Score: {:.4f})".format(hit['score']))
