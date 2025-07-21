import streamlit as st
import pandas as pd
import torch
from PIL import Image
import re
import os
from modules.embedding_utils import compute_clip_similarity, load_clip_model
from modules.image_tools import (
    extract_image_features, select_image_from_text,
    download_image_from_url, compute_clip_image_embedding
)
from modules.translation_utils import translate_query
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np

st.set_page_config(page_title="CIB Coordination Dashboard", layout="wide")
st.title("🕵️ Coordinated Influence Operations Dashboard")

# Initialize CLIP model in session state
if 'clip_model' not in st.session_state:
    st.session_state.clip_model, st.session_state.clip_preprocess = load_clip_model()

# --- Sidebar ---
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

# --- Utility Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def extract_image_urls(df):
    url_cols = ['media', 'media_url', 'attachments', 'media_links']
    image_urls = []
    for _, row in df.iterrows():
        for col in url_cols:
            val = row.get(col, '')
            if isinstance(val, str) and any(x in val for x in ['.jpg', '.png', '.jpeg']):
                image_urls.append(val)
                break
        else:
            image_urls.append(None)
    df['image_url'] = image_urls
    return df

@st.cache_data(show_spinner=False)
def get_clip_embeddings(df):
    embeddings = []
    for url in df['image_url']:
        if url:
            image = download_image_from_url(url)
            if image:
                emb = compute_clip_image_embedding(
                    image, st.session_state.clip_model, st.session_state.clip_preprocess
                )
                embeddings.append(emb)
            else:
                embeddings.append(None)
        else:
            embeddings.append(None)
    df['image_embedding'] = embeddings
    return df

def cluster_image_embeddings(df, eps=0.15, min_samples=2):
    image_vecs = [e for e in df['image_embedding'] if e is not None]
    index_map = [i for i, e in enumerate(df['image_embedding']) if e is not None]
    if not image_vecs:
        df['image_cluster'] = -1
        return df
    embeddings = np.vstack(image_vecs)
    sim_matrix = cosine_similarity(embeddings)
    clustering = DBSCAN(metric='precomputed', eps=1 - eps, min_samples=min_samples)
    labels = clustering.fit_predict(1 - sim_matrix)
    cluster_labels = [-1] * len(df)
    for idx, label in zip(index_map, labels):
        cluster_labels[idx] = label
    df['image_cluster'] = cluster_labels
    return df

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📸 Visual Coordination", "📎 About"])

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

        combined_df['text'] = combined_df['text'].astype(str).apply(clean_text)
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce')

        combined_df = extract_image_urls(combined_df)
        st.session_state['combined_df'] = combined_df

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

        csv_export = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Processed Data", csv_export, "processed_data.csv", "text/csv")

with tab2:
    st.subheader("📸 Visual Coordination Detection")
    combined_df = st.session_state.get('combined_df', pd.DataFrame())
    if not combined_df.empty and 'image_url' in combined_df.columns:
        with st.spinner("Processing media and computing visual embeddings..."):
            combined_df = get_clip_embeddings(combined_df)
            combined_df = cluster_image_embeddings(combined_df, eps=1 - similarity_threshold)

        cluster_counts = combined_df['image_cluster'].value_counts()
        coordinated_clusters = cluster_counts[cluster_counts > 1].index.tolist()

        if coordinated_clusters:
            st.success(f"Found {len(coordinated_clusters)} visually coordinated clusters!")
            for cluster_id in coordinated_clusters:
                st.markdown(f"### Cluster {cluster_id}")
                cluster_df = combined_df[combined_df['image_cluster'] == cluster_id]
                for _, row in cluster_df.iterrows():
                    if row['image_url']:
                        st.image(row['image_url'], width=200, caption=row.get('text', ''))
                        st.caption(f"{row.get('Timestamp')} | {row.get('Source')}")
        else:
            st.info("No visually coordinated image clusters found.")
    else:
        st.warning("Please upload social media data with image URLs.")

with tab3:
    st.markdown("""
    ### 🔍 Overview

    This dashboard helps data journalists investigate coordinated influence behavior across social platforms.

    **Features:**
    - Upload & merge multi-platform datasets
    - Text & CLIP-based visual similarity
    - Multilingual query translation
    - Image-to-image clustering and visual coordination detection

    **Inspired by Vera.AI & Powered by OpenAI CLIP + Streamlit**
    """)
