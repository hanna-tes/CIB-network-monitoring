import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import streamlit as st

# Load Sentence Transformer model once
@st.cache_resource(show_spinner=False)
def load_text_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load CLIP model and processor once
@st.cache_resource(show_spinner=False)
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def get_text_embedding(text):
    """
    Compute embedding for a given text using SentenceTransformer.
    """
    model = load_text_model()
    embedding = model.encode([text])[0]
    return embedding

def compute_text_similarity(text1, text2):
    """
    Compute cosine similarity between two text strings.
    """
    emb1 = get_text_embedding(text1)
    emb2 = get_text_embedding(text2)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return similarity

def batch_text_embeddings(texts):
    """
    Compute embeddings for a list of texts.
    """
    model = load_text_model()
    return model.encode(texts, show_progress_bar=False)

def compute_clip_similarity(image: Image.Image, text: str) -> float:
    """
    Compute similarity between an image and text using CLIP.
    """
    model, processor = load_clip_model()
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    return probs[0][0].item()

def encode_image(image: Image.Image):
    """
    Encode image using CLIP for vector comparison.
    """
    model, processor = load_clip_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

def encode_text_clip(text: str):
    """
    Encode text using CLIP for vector comparison with images.
    """
    model, processor = load_clip_model()
    inputs = processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features / text_features.norm(p=2, dim=-1, keepdim=True)

def clip_image_text_cosine(image: Image.Image, text: str) -> float:
    """
    Compute cosine similarity between CLIP text and image embeddings.
    """
    img_emb = encode_image(image)
    txt_emb = encode_text_clip(text)
    similarity = torch.nn.functional.cosine_similarity(img_emb, txt_emb)
    return similarity.item()
