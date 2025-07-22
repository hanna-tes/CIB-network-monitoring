import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Custom modules
from modules.embedding_utils import compute_text_similarity
from modules.clustering_utils import cluster_texts, build_user_interaction_graph

st.set_page_config(page_title="CIB Dashboard", layout="wide")
st.title("🕵️ CIB Network Monitoring Dashboard")

# --- Sidebar: File Upload ---
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

@st.cache_data(show_spinner=False)
def load_default_dataset():
    url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/Togo_OR_Lome%CC%81_OR_togolais_OR_togolaise_AND_manifest%20-%20Jul%207%2C%202025%20-%205%2012%2053%20PM.csv"
    return pd.read_csv(url, encoding='utf-16', sep='\t', low_memory=False)

# --- Column Mapping ---
col_map = {
    'Influencer': 'Source',
    'Hit Sentence': 'text',
    'Date': 'Timestamp',
    'createTimeISO': 'Timestamp',
    'authorMeta/name': 'Source',
    'message': 'text',
    'title': 'text',
    'media_name': 'Source',
    'channeltitle': 'Source'
}

# --- Platform Detection ---
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

# --- Text Similarity Functions ---
def compute_text_similarity(text1, text2):
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def find_textual_similarities(df, threshold=0.85):
    rows = df[['text', 'Source', 'Timestamp']].dropna()
    similar_pairs = []

    for (i, row1), (j, row2) in combinations(rows.iterrows(), 2):
        score = compute_text_similarity(row1['text'], row2['text'])
        if score >= threshold:
            similar_pairs.append({
                'text1': row1['text'],
                'source1': row1['Source'],
                'time1': row1['Timestamp'],
                'text2': row2['text'],
                'source2': row2['Source'],
                'time2': row2['Timestamp'],
                'similarity': round(score, 3)
            })
    return pd.DataFrame(similar_pairs)

# --- Load Data ---
df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding='utf-16', sep='\t', low_memory=False)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")
else:
    st.sidebar.info("Using default demo dataset")
    df = load_default_dataset()

# --- Standardize Columns ---
if df is not None and not df.empty:
    df.columns = [col_map.get(col.strip(), col.strip()) for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.rename(columns=str.strip)

    # Ensure required columns exist
    required_cols = ["Source", "Timestamp", "text"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    df = df.dropna(subset=['text'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    if 'URL' in df.columns and 'Platform' not in df.columns:
        df['Platform'] = df['URL'].apply(infer_platform_from_url)

# --- Sidebar: Export ---
st.sidebar.markdown("### 📄 Export Results")
def convert_df(data):
    return data.to_csv(index=False).encode('utf-8')

csv_data = convert_df(df)
st.sidebar.download_button("Download Full Data", csv_data, "processed_data.csv", "text/csv")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1 ====================
with tab1:
    st.subheader("📌 Summary Statistics")

    st.markdown("**Top Sources (by post count):**")
    top_sources = df['Source'].value_counts().head(10)
    st.bar_chart(top_sources)

    st.markdown("**Top Hashtags (by frequency):**")
    df['hashtags'] = df['text'].str.findall(r"#\\w+")
    all_hashtags = pd.Series([tag for tags in df['hashtags'] for tag in tags])
    if not all_hashtags.empty:
        st.bar_chart(all_hashtags.value_counts().head(10))
    else:
        st.info("No hashtags found in the dataset.")

    st.markdown("**Posting Activity Over Time**")
    time_series = df.set_index('Timestamp').resample('D').size()
    fig = px.area(time_series, title="Daily Post Volume", labels={'value': 'Posts', 'Timestamp': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2 ====================
with tab2:
    st.subheader("🧠 Similarity & Coordination Detection")

    st.markdown("This table displays groups of posts that are textually similar, possibly indicating coordinated messaging.")
    try:
        text_sim_df = find_textual_similarities(df)
        if not text_sim_df.empty:
            st.dataframe(text_sim_df[['text1', 'source1', 'time1', 'text2', 'source2', 'time2', 'similarity']])
        else:
            st.info("No significant text similarities found.")
    except Exception as e:
        st.warning(f"⚠️ Similarity computation failed: {e}")

    #st.markdown("This table shows image pairs with visual or visual-textual similarities using CLIP.")
    #if 'image_path' in df.columns:
     #   clip_df = compute_visual_clip_similarity(df)
      #  if not clip_df.empty:
      #      st.dataframe(clip_df[['image1', 'image2', 'clip_score']])
      #  else:
       #     st.info("No visually similar content detected.")

# ==================== TAB 3 ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")

    st.markdown("Accounts have been grouped by coordination patterns (hashtags, URLs, posting behavior). Each cluster may indicate potential coordinated activity.")
    try:
        clustered_df = cluster_texts(df)
        cluster_counts = clustered_df['cluster'].value_counts()
        st.dataframe(clustered_df[['Source', 'text', 'Timestamp', 'cluster']])
    except Exception as e:
        st.warning(f"⚠️ Clustering failed: {e}")

    st.markdown("This network graph shows user interactions, clustered by behavioral similarities. Color represents different coordination clusters.")
    try:
        G, pos, cluster_map = build_user_interaction_graph(df)
        fig, ax = plt.subplots(figsize=(10, 7))
        nx.draw(G, pos, with_labels=True, node_color=[cluster_map.get(n, 0) for n in G.nodes], cmap=plt.cm.Set3, node_size=500, edge_color="gray", ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Network graph failed: {e}")
