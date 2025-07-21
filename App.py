import streamlit as st
import pandas as pd
import torch
from PIL import Image
import os
import re
import tempfile
from modules.embedding_utils import compute_clip_similarity, load_clip_model
from modules.image_tools import extract_image_features, select_image_from_text
from modules.translation_utils import translate_query

# Initialize session state
if 'clip_model' not in st.session_state:
    st.session_state.clip_model, st.session_state.clip_preprocess = load_clip_model()

st.set_page_config(page_title="CIB Coordination Dashboard", layout="wide")

st.title("🕵️ Coordinated Influence Operations Dashboard")

# --- Sidebar: Upload and Settings ---
st.sidebar.header("📂 Upload Social Media Files")
uploaded_files = st.sidebar.file_uploader("Upload multiple CSV/Excel files", type=['csv', 'xlsx'], accept_multiple_files=True)

st.sidebar.markdown("---")
similarity_threshold = st.sidebar.slider("🔍 Similarity Threshold", 0.0, 1.0, 0.75, 0.01)
exact_match_toggle = st.sidebar.checkbox("✅ Exact Text Match Only", value=False)

st.sidebar.markdown("---")
st.sidebar.header("🌐 Multilingual Query")
user_query = st.sidebar.text_input("Enter your query (any language)")
use_translation = st.sidebar.checkbox("Translate Query to English", value=True)
translated_query = translate_query(user_query) if use_translation and user_query else user_query

# --- Main Panel ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 CLIP Visual/Text Similarity", "📎 About"])

with tab1:
    st.subheader("1. Upload & Preprocess Data")

    if uploaded_files:
        dfs = []
        for file in uploaded_files:
            if file.name.endswith('.csv'):
                dfs.append(pd.read_csv(file))
            else:
                dfs.append(pd.read_excel(file))
        combined_df = pd.concat(dfs, ignore_index=True)

        # Basic preprocessing
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[@#]\w+", "", text)
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            return text

        combined_df['text'] = combined_df['text'].astype(str).apply(clean_text)
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce')

        st.success("✅ Data loaded and cleaned!")
        st.dataframe(combined_df.head())

        st.subheader("2. Summary Statistics")
        st.markdown(f"**Total Posts:** {len(combined_df)}")
        st.markdown(f"**Unique Users:** {combined_df['Source'].nunique()}")
        st.markdown(f"**Time Range:** {combined_df['Timestamp'].min()} → {combined_df['Timestamp'].max()}")

        if 'hashtags' in combined_df.columns:
            hashtags_series = combined_df['hashtags'].dropna().astype(str).str.split()
            all_hashtags = [tag for sublist in hashtags_series for tag in sublist]
            top_hashtags = pd.Series(all_hashtags).value_counts().head(10)
            st.markdown("**Top Hashtags**")
            st.bar_chart(top_hashtags)

with tab2:
    st.subheader("Visual ↔ Text Similarity Search (via CLIP)")

    uploaded_image = st.file_uploader("Upload an image (e.g., meme or screenshot)", type=["png", "jpg", "jpeg"])

    if uploaded_image and not combined_df.empty:
        with st.spinner("Computing image-text similarity..."):
            image = Image.open(uploaded_image).convert("RGB")
            texts = combined_df['text'].tolist()

            top_texts = compute_clip_similarity(image, texts, st.session_state.clip_model, st.session_state.clip_preprocess, top_k=5)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown("**Top Matching Text Posts**")
        for score, match_text in top_texts:
            st.markdown(f"**Score:** `{score:.4f}`\n\n>{match_text}")

    elif uploaded_image:
        st.warning("Please upload social media data first in the Dashboard tab.")

with tab3:
    st.markdown("""
    ### 🔍 Overview

    This dashboard is designed to help investigate coordinated influence behavior across multiple social media platforms.

    **Features:**
    - Upload & merge multi-platform datasets
    - CLIP-based visual-to-text similarity detection
    - Multilingual query translation
    - Similarity slider + exact-match toggle

    **Powered by**: OpenAI CLIP, HuggingFace Transformers, Streamlit
    """)

