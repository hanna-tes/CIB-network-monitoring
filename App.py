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
st.title("🕵️ CIB Network Monitoring Dashboard")

# --- Sidebar: File Upload ---
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Replace your load_default_dataset function with this:

@st.cache_data(show_spinner=False)
def load_default_dataset():
    url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/your_file.csv"  # Replace with your actual raw CSV URL
    return pd.read_csv(url, encoding='utf-16', sep='\t', low_memory=False)

# When handling uploaded files:

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

# --- Column Standardization ---
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
df.columns = [col_map.get(col, col) for col in df.columns]
df = df.rename(columns=str.strip)
df = df.dropna(subset=['text'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# --- Sidebar: Export ---
st.sidebar.markdown("### 📤 Export Results")
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
