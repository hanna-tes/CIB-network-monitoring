import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import streamlit as st

# Load models only once
@st.cache_resource
def load_text_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def compute_text_similarity(texts):
    model = load_text_model()
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def compute_clip_similarity(images, texts):
    """
    images: list of PIL Images
    texts: list of strings
    """
    model, processor = load_clip_model()
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape: [batch_size, text_count]
    probs = logits_per_image.softmax(dim=1)
    return probs.detach().numpy()
