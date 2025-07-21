import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
import os
import re
import tempfile
import networkx as nx
import plotly.express as px
import altair as alt
from datetime import timedelta
from modules.embedding_utils import compute_clip_similarity, load_clip_model, generate_image_embeddings
from modules.image_tools import extract_image_urls
from modules.clustering_utils import cluster_embeddings

# Initialize session state
if 'clip_model' not in st.session_state:
    st.session_state.clip_model, st.session_state.clip_preprocess = load_clip_model()

st.set_page_config(page_title="CIB Coordination Dashboard", layout="wide")
st.title("🕵️ Coordinated Inauthentic Behavior (CIB) Dashboard")

# --- Sidebar: Upload and Settings ---
st.sidebar.header("📂 Upload Social Media Files")
uploaded_files = st.sidebar.file_uploader("Upload multiple CSV/Excel files", type=['csv', 'xlsx'], accept_multiple_files=True)

st.sidebar.markdown("---")
similarity_threshold = st.sidebar.slider("🔍 Similarity Threshold (Visual/Text)", 0.0, 1.0, 0.75, 0.01)
exact_match_toggle = st.sidebar.checkbox("✅ Exact Text Match Only", value=False)

# --- Data Preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        if file.name.endswith('.csv'):
            dfs.append(pd.read_csv(file))
        else:
            dfs.append(pd.read_excel(file))
    combined_df = pd.concat(dfs, ignore_index=True)

    if 'text' in combined_df.columns:
        combined_df['text'] = combined_df['text'].astype(str).apply(clean_text)

    if 'Timestamp' in combined_df.columns:
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce')

    st.session_state['combined_df'] = combined_df

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 CLIP Visual Coordination", "📎 About"])

with tab1:
    st.subheader("1. Upload & Explore Data")

    combined_df = st.session_state.get('combined_df', pd.DataFrame())

    if not combined_df.empty:
        st.success("✅ Data loaded and cleaned!")
        st.dataframe(combined_df.head())

        st.subheader("2. Summary Statistics")
        st.markdown(f"**Total Posts:** {len(combined_df)}")
        if 'Source' in combined_df.columns:
            st.markdown(f"**Unique Users:** {combined_df['Source'].nunique()}")
        if 'Timestamp' in combined_df.columns:
            st.markdown(f"**Time Range:** {combined_df['Timestamp'].min()} → {combined_df['Timestamp'].max()}")

        if 'hashtags' in combined_df.columns:
            hashtags_series = combined_df['hashtags'].dropna().astype(str).str.split()
            all_hashtags = [tag for sublist in hashtags_series for tag in sublist]
            top_hashtags = pd.Series(all_hashtags).value_counts().head(10)
            st.markdown("**Top Hashtags**")
            st.bar_chart(top_hashtags)

        if 'Timestamp' in combined_df.columns:
            st.subheader("3. Posting Timeline")
            timeline = combined_df.copy()
            timeline['hour'] = timeline['Timestamp'].dt.floor('h')
            post_counts = timeline.groupby('hour').size().reset_index(name='counts')
            chart = alt.Chart(post_counts).mark_line().encode(
                x='hour:T',
                y='counts:Q'
            ).properties(title='Posting Activity Over Time', width=700)
            st.altair_chart(chart)

        if 'Source' in combined_df.columns:
            st.subheader("4. Most Active Users")
            top_users = combined_df['Source'].value_counts().head(10)
            st.bar_chart(top_users)

        if 'urls' in combined_df.columns:
            st.subheader("5. Coordinated URL Sharing (Fast Reposts)")
            url_posts = combined_df.dropna(subset=['urls', 'Timestamp'])
            suspicious_urls = []
            for url, group in url_posts.groupby('urls'):
                group = group.sort_values('Timestamp')
                deltas = group['Timestamp'].diff().dropna()
                rapid_posts = (deltas <= timedelta(minutes=5)).sum()
                if rapid_posts >= 2:
                    suspicious_urls.append(url)
            if suspicious_urls:
                st.markdown("**Potentially Coordinated URLs:**")
                for u in suspicious_urls:
                    st.write(f"🔗 {u}")
            else:
                st.write("No coordinated reposts detected based on timing.")

with tab2:
    st.subheader("Visual ↔ Text Similarity Detection (via CLIP)")
    combined_df = st.session_state.get('combined_df', pd.DataFrame())

    if not combined_df.empty:
        st.markdown("### 🖼️ Extracting and Embedding Images from Posts")

        image_urls = extract_image_urls(combined_df)
        if not image_urls:
            st.warning("No image URLs found in posts.")
        else:
            with st.spinner("Downloading and embedding images..."):
                image_embeddings, valid_images = generate_image_embeddings(image_urls, st.session_state.clip_model, st.session_state.clip_preprocess)

            st.markdown(f"**{len(valid_images)}** images embedded.")

            if len(image_embeddings) >= 2:
                clusters = cluster_embeddings(image_embeddings, eps=0.3)
                clustered = pd.DataFrame({'image': valid_images, 'cluster': clusters})

                st.subheader("📸 Detected Visual Clusters")
                for cluster_id in sorted(clustered['cluster'].unique()):
                    cluster_imgs = clustered[clustered['cluster'] == cluster_id]['image'].tolist()
                    st.markdown(f"**Cluster {cluster_id}** ({len(cluster_imgs)} similar images)")
                    st.image(cluster_imgs, width=150)

with tab3:
    st.markdown("""
    ### 🔍 Dashboard Overview

    This tool supports detection of **Coordinated Inauthentic Behavior (CIB)** across multiple social platforms.

    **Key Features:**
    - Upload & preprocess CSV/Excel data
    - Merge multiple datasets
    - CLIP-based visual coordination detection (image clusters)
    - Shared URL repost detection
    - Posting spikes & timeline charts
    - Top users and hashtags
    - Text/image coordination similarity search

    **Coming Soon:**
    - User interaction networks
    - Exportable reports and graphs
    - Hashtag + content overlap coordination

    🧠 Powered by: OpenAI CLIP · HuggingFace · Streamlit
    """)
