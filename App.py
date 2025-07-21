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
from modules.embedding_utils import compute_text_similarity
from modules.clustering_utils import cluster_texts, build_user_interaction_graph

# Initialize session state
st.set_page_config(page_title="CIB Monitoring Dashboard", layout="wide")
st.title("🕵️ CIB monitoring and analysis Dashboard")

# --- Sidebar: Upload and Settings ---
st.sidebar.header("📂 Upload Social Media Files")
uploaded_files = st.sidebar.file_uploader("Upload multiple CSV/Excel files", type=['csv', 'xlsx'], accept_multiple_files=True)

st.sidebar.markdown("---")
similarity_threshold = st.sidebar.slider("🔍 Similarity Threshold (Text Only)", 0.0, 1.0, 0.75, 0.01)
exact_match_toggle = st.sidebar.checkbox("✅ Exact Text Match Only", value=False)

# --- Data Preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

combined_df = pd.DataFrame()

default_url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/Togo_OR_Lome%CC%81_OR_togolais_OR_togolaise_AND_manifest%20-%20Jul%207%2C%202025%20-%205%2012%2053%20PM.csv"

try:
    combined_df = pd.read_csv(default_url, encoding='utf-16', on_bad_lines='skip',sep='\t',low_memory=False)
except Exception as e:
    st.warning(f"Failed to load default dataset: {e}")

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        if file.name.endswith('.csv'):
            dfs.append(pd.read_csv(file, encoding='utf-16', on_bad_lines='skip',sep='\t',low_memory=False))
        else:
            dfs.append(pd.read_excel(file))
    combined_df = pd.concat(dfs, ignore_index=True)

if not combined_df.empty:
    if 'text' in combined_df.columns:
        combined_df['text'] = combined_df['text'].astype(str).apply(clean_text)

    if 'Timestamp' in combined_df.columns:
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce')

    st.session_state['combined_df'] = combined_df

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Text Similarity", "📎 About"])

with tab1:
    st.subheader("1. Upload & Explore Data")
    combined_df = st.session_state.get('combined_df', pd.DataFrame())

    if not combined_df.empty:
        st.success("✅ Data loaded and cleaned!")
        st.dataframe(combined_df.head())

        st.subheader("2. Summary Statistics")
        st.markdown(f"**Total Posts:** {len(combined_df)}")
        if 'Source' in combined_df.columns:
            st.markdown(f"**Unique Sources:** {combined_df['Source'].nunique()}")

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
            st.subheader("4. Most Active Sources")
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

        if 'text' in combined_df.columns:
            st.subheader("6. Text Clustering")
            labels = cluster_texts(combined_df['text'].tolist())
            combined_df['cluster'] = labels
            fig = px.histogram(combined_df, x='cluster')
            st.plotly_chart(fig)

        if 'Source' in combined_df.columns:
            st.subheader("7. User Interaction Network (based on Source)")
            G = build_user_interaction_graph(combined_df)
            st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

with tab2:
    st.subheader("Text Similarity Detection")
    combined_df = st.session_state.get('combined_df', pd.DataFrame())

    if not combined_df.empty and 'text' in combined_df.columns:
        with st.spinner("Computing text similarity..."):
            similar_pairs = compute_text_similarity(combined_df['text'].tolist(), threshold=similarity_threshold, exact=exact_match_toggle)

        if similar_pairs:
            st.markdown(f"**Detected {len(similar_pairs)} Similar Text Pairs:**")
            for idx1, idx2, score in similar_pairs:
                st.markdown(f"**Post {idx1}** ↔ **Post {idx2}** (Score: {score:.2f})")
                st.text_area("Post 1", combined_df.iloc[idx1]['text'], height=80)
                st.text_area("Post 2", combined_df.iloc[idx2]['text'], height=80)
                st.markdown("---")
        else:
            st.info("No similar text posts found at the selected threshold.")

with tab3:
    st.markdown("""
    ### 🔍 Dashboard Overview

    This tool supports detection of **Coordinated Inauthentic Behavior (CIB)** across multiple social platforms.

    **Key Features:**
    - Upload & preprocess CSV/Excel data
    - Merge multiple datasets
    - Posting timeline and top users
    - Shared URL repost detection
    - Text-based similarity detection
    - Now includes text clustering & source interaction graph

    🧠 Powered by: OpenAI · HuggingFace · Streamlit
    """)
