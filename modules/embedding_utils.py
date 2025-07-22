import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import streamlit as st

# Load models once and cache
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource
def load_text_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

clip_model, clip_processor = load_clip_model()
text_model = load_text_model()


# Text embeddings
@st.cache_data(show_spinner=False)
def get_text_embedding(text):
    embedding = text_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    return embedding.cpu().numpy()



# Compute cosine similarity
def compute_text_similarity(text1, text2):
    emb1 = get_text_embedding(text1)
    emb2 = get_text_embedding(text2)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return similarity


