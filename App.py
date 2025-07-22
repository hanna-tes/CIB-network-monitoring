# App.py 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import timedelta
import os

# Try to import custom modules with graceful fallbacks
try:
    from modules.embedding_utils import compute_text_similarity, compute_visual_clip_similarity
except ImportError as e:
    st.warning(f"⚠️ Failed to load embedding utils: {e}")

    def compute_text_similarity(df):
        return pd.DataFrame()

    def compute_visual_clip_similarity(df):
        return pd.DataFrame()

try:
    from modules.clustering_utils import cluster_texts, build_user_interaction_graph
except ImportError as e:
    st.warning(f"⚠️ Failed to load clustering utils: {e}")

    def cluster_texts(df):
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy

    def build_user_interaction_graph(df):
        G = nx.Graph()
        sources = df['Source'].dropna().unique()[:10]
        for src in sources:
            G.add_node(src)
        return G, {}, {}

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="CIB Dashboard", layout="wide")
st.title("🕵️ CIB Network Monitoring Dashboard")

# -------------------------------
# Sidebar: File Upload
# -------------------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file", 
    type=["csv", "xlsx"]
)

# -------------------------------
# Load Data: Uploaded or Default (from GitHub to avoid local path issues)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_default_dataset():
    default_url = (
        "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/ "
        "refs/heads/main/Togo_OR_Lome%CC%81_OR_togolais_OR_togolaise_AND_manifest%20-%20Jul%207%2C%202025%20-%205%2012%2053%20PM.csv"
    )
    try:
        df = pd.read_csv(default_url, encoding='utf-16', sep='\t', on_bad_lines='skip', low_memory=False)
        st.info("✅ Loaded default dataset from GitHub.")
        return df
    except Exception as e:
        st.error(f"❌ Could not load default dataset: {e}")
        return pd.DataFrame()

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded: `{uploaded_file.name}`")
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()
else:
    st.sidebar.info("ℹ️ No upload detected. Using default dataset.")
    df = load_default_dataset()

if df is None or df.empty:
    st.error("❌ No data available. Please upload a file or check the default dataset connection.")
    st.stop()

# -------------------------------
# Column Standardization
# -------------------------------

# Clean column names: strip whitespace
df.columns = [str(col).strip() for col in df.columns]

# Define mapping
col_map = {
    # Source
    'Influencer': 'Source',
    'authorMeta/name': 'Source',
    'media_name': 'Source',
    'channeltitle': 'Source',
    'from_name': 'Source',

    # Text
    'Hit Sentence': 'text',           # ← Priority over Opening Text
    'message': 'text',
    'title': 'text',
    'Opening Text': '_DROP_',         # Explicitly ignore

    # Timestamp
    'Date': 'Timestamp',
    'date': 'Timestamp',
    'createTimeISO': 'Timestamp',

    # URL
    'url': 'URL',
    'URL': 'URL',                    # Already correct
    'webVideoUrl': 'URL',
    'post_url': 'URL',
    'link': 'URL'
}

# Apply mapping only to existing columns, ignoring _DROP_
rename_dict = {}
for col, std in col_map.items():
    if col in df.columns and std != '_DROP_':
        rename_dict[col] = std

# Also detect any variation of "URL"
for col in df.columns:
    if col.lower() in ['url', 'urls', 'link', 'links']:
        rename_dict[col] = 'URL'

# Rename
df.rename(columns=rename_dict, inplace=True)

# Ensure required columns exist
for col in ['Source', 'text', 'Timestamp']:
    if col not in df.columns:
        st.error(f"❌ Missing required column: `{col}`")
        st.stop()

# Drop rows with missing text or timestamp
df = df.dropna(subset=['text']).reset_index(drop=True)

# Parse Timestamp safely
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='ISO8601')
    df = df.dropna(subset=['Timestamp']).reset_index(drop=True)

# Create hashtags
df['hashtags'] = df['text'].astype(str).str.findall(r"#\w+")

# Add Platform detection
if 'authorMeta/name' in [c.strip() for c in uploaded_file.name for uploaded_file in [uploaded_file]] if uploaded_file else [] or \
   ('webVideoUrl' in df.columns and 'TikTok' in df.get('Platform', '')):
    df['Platform'] = 'TikTok'
elif 'channeltitle' in df.columns:
    df['Platform'] = 'Telegram'
elif 'media_name' in df.columns:
    df['Platform'] = 'Media'
else:
    df['Platform'] = 'X'

# -------------------------------
# Sidebar: Export Processed Data
# -------------------------------
st.sidebar.markdown("### 📤 Export Results")
@st.cache_data
def convert_df(data):
    return data.to_csv(index=False).encode('utf-8')

csv_data = convert_df(df)
st.sidebar.download_button(
    label="📥 Download Full Data",
    data=csv_data,
    file_name="cib_processed_data.csv",
    mime="text/csv"
)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.subheader("📌 Summary Statistics")

    # Top Sources
    st.markdown("**Top Sources (by post count):**")
    top_sources = df['Source'].value_counts().head(10)
    st.bar_chart(top_sources)

    # Top Hashtags
    st.markdown("**Top Hashtags (by frequency):**")
    all_hashtags = [tag for tags in df['hashtags'] for tag in tags]
    if all_hashtags:
        hashtag_freq = pd.Series(all_hashtags).value_counts().head(10)
        st.bar_chart(hashtag_freq)
    else:
        st.info("No hashtags found.")

    # Posting Over Time
    st.markdown("**Posting Activity Over Time:**")
    df['date_only'] = pd.to_datetime(df['Timestamp']).dt.date
    time_series = df.groupby('date_only').size()
    st.line_chart(time_series)

# ==================== TAB 2: ANALYSIS ====================
with tab2:
    st.subheader("🧠 Similarity & Coordination Detection")

    # Text Similarity
    st.markdown("### 🔤 Text Similarity")
    st.markdown("Posts with high semantic similarity may indicate coordinated messaging.")
    try:
        sim_df = compute_text_similarity(df)
        if not sim_df.empty and len(sim_df) > 0:
            st.dataframe(sim_df[
                ['text1', 'source1', 'time1', 'text2', 'source2', 'time2', 'similarity']
            ])
        else:
            st.info("No significant textual similarities found.")
    except Exception as e:
        st.warning(f"⚠️ Text similarity computation failed: {e}")

    # Visual Similarity (CLIP)
    #if 'image_path' in df.columns or 'image_url' in df.columns:
       # st.markdown("### 🖼️ Visual Similarity (CLIP)")
       # st.markdown("Detecting visually similar content using CLIP embeddings.")
       # try:
       #     clip_df = compute_visual_clip_similarity(df)
        #    if not clip_df.empty:
                st.dataframe(clip_df[['image1', 'image2', 'clip_score']])
       #     else:
                st.info("No visually similar pairs detected.")
      #  except Exception as e:
     #       st.warning(f"⚠️ Visual similarity failed: {e}")
   # else:
    #    st.markdown("📷 No image paths provided — skipping visual analysis.")

# ==================== TAB 3: NETWORK & RISK ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")

    # Clustering
    st.markdown("### 🧩 Behavioral Clustering")
    st.markdown("Users grouped by shared language, hashtags, or posting patterns.")
    try:
        clustered_df = cluster_texts(df)
        if 'cluster' in clustered_df.columns:
            fig = px.histogram(clustered_df, x='cluster', title="Distribution of Clusters")
            st.plotly_chart(fig)
            st.dataframe(clustered_df[['Source', 'text', 'Timestamp', 'cluster']].head(10))
        else:
            st.info("Clustering completed but no cluster labels returned.")
    except Exception as e:
        st.warning(f"⚠️ Clustering failed: {e}")

    # Network Graph
    st.markdown("### 🌐 User Interaction Network")
    st.markdown("Nodes represent users; edges show interactions or behavioral similarity.")
    try:
        G, pos, cluster_map = build_user_interaction_graph(df)
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.cm.tab20 if max(cluster_map.values(), default=1) > 10 else plt.cm.Set3
        nx.draw(
            G, pos,
            with_labels=True,
            node_color=[cluster_map.get(n, 0) for n in G.nodes],
            cmap=cmap,
            node_size=600,
            edge_color="lightgray",
            font_size=10,
            ax=ax
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Network graph could not be generated: {e}")

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
