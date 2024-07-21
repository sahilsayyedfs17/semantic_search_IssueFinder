import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
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
    for i in range(0, len(corpus), batch_size):
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

# Function to save feedback
def save_feedback(query, selected_responses):
    feedback_data = []
    for response in selected_responses:
        feedback_data.append(f"{query},{response},1")  # Assume feedback is positive (1) for selected responses

    with open("feedback.csv", "a") as f:
        for entry in feedback_data:
            f.write(f"{entry}\n")

# Streamlit UI
st.title("Similar Issue Finder")

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
    top_k = st.slider("Number of similar issues to retrieve:", 1, 100, 10)

    if 'hits' not in st.session_state:
        st.session_state.hits = None

    if st.button("Find Similar Issues"):
        st.write("Finding similar issues...")
        query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
        st.session_state.hits = hits[0]  # Get the hits for the first query

    if st.session_state.hits:
        hits = st.session_state.hits
        st.write(f"\nTop {top_k} most similar issues in corpus:")

        # Initialize session state for checkboxes
        if 'checkbox_states' not in st.session_state:
            st.session_state.checkbox_states = {}

        responses = []
        for hit in hits:
            hit_id = hit['corpus_id']
            article_data = df.iloc[hit_id]
            title = article_data["Issue Key"] + " -> " + article_data["Summary"]
            responses.append(title)

            # Use a key to preserve the state of each checkbox
            checkbox_key = f"checkbox_{hit_id}"
            if checkbox_key not in st.session_state.checkbox_states:
                st.session_state.checkbox_states[checkbox_key] = False

            # Display checkbox and update state
            st.session_state.checkbox_states[checkbox_key] = st.checkbox(
                f"{title} (Score: {hit['score']:.4f})", 
                value=st.session_state.checkbox_states[checkbox_key], 
                key=checkbox_key
            )

        if st.button("Submit Feedback"):
            selected_responses = [response for response, checked in zip(responses, st.session_state.checkbox_states.values()) if checked]
            if selected_responses:
                save_feedback(query, selected_responses)
                st.write("Thank you for your feedback!")
                st.write("Feedback saved successfully!")
            else:
                st.write("No responses selected for feedback.")

        # Debugging output to ensure state management
        st.write(f"Current Checkbox States: {st.session_state.checkbox_states}")
