import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np

def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    return model, preprocess

def compute_clip_similarity(image, texts, model, preprocess, top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [(similarities[i], texts[i]) for i in top_indices]

def generate_image_embeddings(image_urls, model, preprocess):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = []
    valid_images = []

    for url in image_urls:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(img_tensor)
                emb /= emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy()[0])
                valid_images.append(img)
        except Exception as e:
            print(f"Failed to process {url}: {e}")

    return np.array(embeddings), valid_images
