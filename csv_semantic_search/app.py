from sentence_transformers import SentenceTransformer, util
import pandas as pd

df = pd.read_csv("cleaned_VTVAS.csv")
print(len(df)) # 192368

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Concatenate titles and article texts
corpus = df["Description"] + "." + df["Summary"]

# Create embeddings from titles and texts of articles
# It takes about 10 minutes for 192368 articles
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


query = 'no pic is shown on tiles'
top_k = 10

# Find the closest top_k sentences of the corpus based on cosine similarity
query_embedding = embedder.encode(query, convert_to_tensor=True)
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
hits = hits[0] # Get the hits for the first query

print(f"\nTop {top_k} most similar sentences in corpus:")
for hit in hits:
    hit_id = hit['corpus_id']
    article_data = df.iloc[hit_id]
    title = article_data["Summary"]
    print("-", title, "(Score: {:.4f})".format(hit['score']))