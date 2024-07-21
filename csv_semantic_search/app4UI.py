import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
import torch
import os

# Check if GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.to(device)

embedder = load_model()

# Function to encode with progress bar
@st.cache_data(show_spinner=False, max_entries=1)
def encode_with_progress(corpus, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="Encoding sentences"):
        batch = corpus[i:i+batch_size]
        batch_embeddings = embedder.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Function to load and preprocess data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Description'] = df['Description'].fillna('').astype(str)
    df['Summary'] = df['Summary'].fillna('').astype(str)
    corpus = (df["Description"] + ". " + df["Summary"]).tolist()
    return df, corpus

# Streamlit UI
st.title("Sentence Similarity Finder")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df, corpus = load_and_preprocess_data(uploaded_file)
    
    # Check if embeddings file exists
    embeddings_file_path = "corpus_embeddings.pt"
    if os.path.exists(embeddings_file_path):
        st.write("Loading existing embeddings...")
        corpus_embeddings = torch.load(embeddings_file_path)
    else:
        st.write("Creating embeddings...")
        corpus_embeddings = encode_with_progress(corpus)
        st.write("Saving embeddings...")
        torch.save(corpus_embeddings, embeddings_file_path)

    # Query input
    query = st.text_input("Enter your query:")
    top_k = st.slider("Number of similar sentences to retrieve:", 1, 100, 10)

    if st.button("Find Similar Sentences"):
        st.write("Finding similar sentences...")
        query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]  # Get the hits for the first query

        st.write(f"\nTop {top_k} most similar sentences in corpus:")
        for hit in hits:
            hit_id = hit['corpus_id']
            article_data = df.iloc[hit_id]
            title = article_data["Issue Key"] + " -> " + article_data["Summary"]
            st.write(f"- {title} (Score: {hit['score']:.4f})")
