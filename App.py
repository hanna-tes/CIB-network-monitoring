import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from datetime import timedelta

# Custom modules
from modules.embedding_utils import compute_text_similarity
from modules.clustering_utils import cluster_texts, build_user_interaction_graph

st.set_page_config(page_title="CIB Dashboard", layout="wide")
st.title("🕵️ Coordinated Inauthentic Behavior (CIB) Network Monitoring Dashboard")

# === File Upload or Use Default ===
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV/TSV file", type=["csv", "tsv"])

# Target columns
#Map various column names to standard ones
col_map = {
    'Influencer': 'Source',
    'authorMeta/name': 'Source',
    'media_name': 'Source',
    'channeltitle': 'Source',
    'Hit Sentence': 'text',
    'message': 'text',
    'title': 'text',
    'Date': 'Timestamp',
    'createTimeISO': 'Timestamp',
}

# Detect platform from URL
def infer_platform_from_url(url):
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    if "tiktok.com" in url:
        return "TikTok"
    elif "facebook.com" in url or "fb.watch" in url:
        return "Facebook"
    elif "twitter.com" in url or "x.com" in url:
        return "Twitter"
    elif "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    elif "instagram.com" in url:
        return "Instagram"
    elif "telegram.me" in url or "t.me" in url:
        return "Telegram"
    elif url.startswith("https://"):
        return "Media"
    else:
        return "Unknown"

# Load and clean dataset
def load_and_standardize_data(df):
    # Standardize column names
    df.columns = [col_map.get(col.strip(), col.strip()) for col in df.columns]
    df = df.rename(columns=str.strip)

    # Required columns
    essential = ["Source", "Timestamp", "text", "URL"]
    missing = [col for col in essential if col not in df.columns]
    if missing:
        st.sidebar.error(f"❌ Missing required columns: {missing}")
        st.stop()

    # Add Platform column if missing
    if "Platform" not in df.columns:
        df["Platform"] = df["URL"].apply(infer_platform_from_url)

    # Subset only relevant columns
    df = df[["Source", "Timestamp", "text", "URL", "Platform"]].copy()

    # Clean timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp", "text"], inplace=True)
    return df

# Load default data from GitHub
@st.cache_data
def load_default_dataset():
    url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/Togo_OR_Lome%CC%81_OR_togolais_OR_togolaise_AND_manifest%20-%20Jul%207%2C%202025%20-%205%2012%2053%20PM.csv"
    try:
        df = pd.read_csv(url, encoding='utf-16', sep='\t', low_memory=False)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load default data: {e}")
        return pd.DataFrame()

# Sidebar upload or default
st.sidebar.title("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv", "tsv"])

if uploaded_file:
    st.sidebar.success("✅ File uploaded")
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-16', sep='\t', low_memory=False)
    except Exception as e:
        st.sidebar.error(f"❌ Failed to read uploaded file: {e}")
        st.stop()
else:
    st.sidebar.warning("⚠️ Using default dataset from GitHub.")
    df = load_default_dataset()

# Standardize & clean data
if not df.empty:
    df = load_and_standardize_data(df)
    st.success(f"✅ Loaded {len(df)} posts from {df['Platform'].nunique()} platforms")
    st.dataframe(df.head(50))
else:
    st.error("❌ No data to display.")

# --- Preview ---
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10))

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1 ====================
with tab1:
    st.subheader("📌 Summary Statistics")

    st.markdown("**Top Sources (by post count):**")
    top_sources = df['Source'].value_counts().head(10)
    st.bar_chart(top_sources)

    st.markdown("**Top Hashtags (by frequency):**")
    df['hashtags'] = df['text'].str.findall(r"#\w+")
    all_hashtags = pd.Series([tag for tags in df['hashtags'] for tag in tags])
    st.bar_chart(all_hashtags.value_counts().head(10))

    st.markdown("**Posting Spikes Over Time**")
    time_series = df.groupby(pd.to_datetime(df['Timestamp']).dt.date).size()
    st.line_chart(time_series)

# ==================== TAB 2 ====================
with tab2:
    st.subheader("🧠 Similarity & Coordination Detection")
    
    # Text Similarity
    st.markdown("This table displays groups of posts that are textually similar, possibly indicating coordinated messaging.")
    try:
        text_sim_df = compute_text_similarity(df)
        if not text_sim_df.empty:
            st.dataframe(text_sim_df[['text1', 'source1', 'time1', 'text2', 'source2', 'time2', 'similarity']])
        else:
            st.info("No significant text similarities found.")
    except Exception as e:
        st.warning(f"⚠️ Similarity computation failed: {e}")

    # Visual Similarity (CLIP)
    #st.markdown("This table shows image pairs with visual or visual-textual similarities using CLIP.")
    #if 'image_path' in df.columns:
     #   clip_df = compute_visual_clip_similarity(df)
      #  if not clip_df.empty:
      #      st.dataframe(clip_df[['image1', 'image2', 'clip_score']])
     #   else:
       #     st.info("No visually similar content detected.")

# ==================== TAB 3 ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")

    # Clustering for High-Risk Account Grouping
    st.markdown("Accounts have been grouped by coordination patterns (hashtags, URLs, posting behavior). Each cluster may indicate potential coordinated activity.")
    try:
        clustered_df = cluster_texts(df)
        cluster_counts = clustered_df['cluster'].value_counts()
        st.dataframe(clustered_df[['Source', 'text', 'Timestamp', 'cluster']])
    except Exception as e:
        st.warning(f"⚠️ Clustering failed: {e}")

    # Network Graph
    st.markdown("This network graph shows user interactions, clustered by behavioral similarities. Color represents different coordination clusters.")
    try:
        G, pos, cluster_map = build_user_interaction_graph(df)
        fig, ax = plt.subplots(figsize=(10, 7))
        nx.draw(G, pos, with_labels=True, node_color=[cluster_map.get(n, 0) for n in G.nodes], cmap=plt.cm.Set3, node_size=500, edge_color="gray", ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Network graph failed: {e}")
