import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import re
import os

# Custom modules (assumed to exist)
from modules.embedding_utils import compute_text_similarity
from modules.clustering_utils import cluster_texts, build_user_interaction_graph

# Set page config
st.set_page_config(page_title="CIB Dashboard", layout="wide")
st.title("🕵️ CIB Network Monitoring Dashboard")

# --- Helper Functions ---
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

def extract_original_text(text):
    """Remove RT @user: prefix to get the core message"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^RT\s+@\w+:\s*', '', text).strip()
    return cleaned

@st.cache_data(show_spinner=False)
def load_default_dataset():
    url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/TogoJULYData%20-%20Sheet1.csv "
    try:
        return pd.read_csv(url, encoding='utf-16', sep='\t', low_memory=False)
    except Exception as e:
        st.error(f"Failed to load default dataset: {e}")
        return pd.DataFrame()

def find_textual_similarities(df, threshold=0.85):
    """Find similar texts using original content (after removing RT)"""
    rows = df[['original_text', 'Source', 'Timestamp']].dropna()
    rows = rows[rows['original_text'] != ""]
    similar_pairs = []
    seen_combinations = set()

    # Convert to list of dicts for faster iteration
    row_list = [(idx, row) for idx, row in rows.iterrows()]

    for i, (idx1, row1) in enumerate(row_list):
        for j, (idx2, row2) in enumerate(row_list):
            if i >= j:
                continue

            t1, t2 = row1['original_text'], row2['original_text']

            # Exact match first (fast)
            if t1 == t2:
                score = 1.0
            else:
                score = compute_text_similarity(t1, t2)

            if score >= threshold:
                key = tuple(sorted([idx1, idx2]))
                if key in seen_combinations:
                    continue
                seen_combinations.add(key)

                similar_pairs.append({
                    'text1': t1,
                    'source1': row1['Source'],
                    'time1': row1['Timestamp'],
                    'text2': row2['original_text'],
                    'source2': row2['Source'],
                    'time2': row2['Timestamp'],
                    'similarity': round(score, 3),
                    'shared_narrative': t1[:150] + ("..." if len(t1) > 150 else "")
                })

    return pd.DataFrame(similar_pairs)

# --- Sidebar: Upload & Filters ---
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Load data
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding='utf-16', sep='\t', low_memory=False)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")
        df = None
else:
    st.sidebar.info("Using default demo dataset")
    df = load_default_dataset()

# Exit if no data
if df is None or df.empty:
    st.warning("No data available. Please upload a valid dataset.")
    st.stop()

# --- Standardize Columns ---
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

df.columns = [col_map.get(col.strip(), col.strip()) for col in df.columns]
df = df.loc[:, ~df.columns.duplicated()]
df = df.rename(columns=str.strip)

required_cols = ["Source", "Timestamp", "text"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

df = df.dropna(subset=['text']).reset_index(drop=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp']).reset_index(drop=True)

# Add Platform if missing
if 'URL' in df.columns and 'Platform' not in df.columns:
    df['Platform'] = df['URL'].apply(infer_platform_from_url)
elif 'Platform' not in df.columns:
    df['Platform'] = "Unknown"

# Preprocess: Extract original text (remove RT)
df['original_text'] = df['text'].apply(extract_original_text)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

platforms = st.sidebar.multiselect(
    "Platforms", 
    options=df['Platform'].unique().tolist(), 
    default=df['Platform'].unique().tolist()
)

# Apply filters
try:
    start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
except:
    start_dt, end_dt = min_date, max_date
filtered_df = df[
    (df['Timestamp'] >= start_dt) &
    (df['Timestamp'] < end_dt) &
    (df['Platform'].isin(platforms))
].copy()

# Export button
st.sidebar.markdown("### 📄 Export Results")
@st.cache_data
def convert_df(data):
    return data.to_csv(index=False).encode('utf-8')

csv_data = convert_df(filtered_df)
st.sidebar.download_button("Download Filtered Data", csv_data, "filtered_data.csv", "text/csv")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Summary Statistics")

    # Top Sources
    st.markdown("**Top Sources (by post count):**")
    top_sources = filtered_df['Source'].value_counts().head(10)
    fig_src = px.bar(top_sources, title="Top 10 Active Accounts", labels={'value': 'Posts', 'index': 'Account'})
    st.plotly_chart(fig_src, use_container_width=True)

    # Hashtags
    st.markdown("**Top Hashtags:**")
    filtered_df['hashtags'] = filtered_df['text'].str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
    all_hashtags = [tag for tags in filtered_df['hashtags'] for tag in tags]
    if all_hashtags:
        hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
        fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags", labels={'value': 'Frequency', 'index': 'Hashtag'})
        st.plotly_chart(fig_ht, use_container_width=True)
    else:
        st.info("No hashtags found.")

    # Time Series
    st.markdown("**Daily Posting Activity**")
    time_series = filtered_df.set_index('Timestamp').resample('D').size()
    fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'Timestamp': 'Date'})
    st.plotly_chart(fig_ts, use_container_width=True)


# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🧠 Narrative Detection & Coordination")

    with st.spinner("Analyzing textual similarities..."):
        try:
            sim_df = find_textual_similarities(filtered_df, threshold=0.85)
            if not sim_df.empty:
                st.success(f"✅ Found {len(sim_df)} pairwise similarities indicating coordination.")

                # Aggregate narratives
                narrative_summary = sim_df.groupby('shared_narrative').agg(
                    share_count=('similarity', 'count'),
                    sources_involved=('source1', lambda x: ", ".join(x.astype(str)[:5]) + ("..." if len(x) > 5 else ""))
                ).sort_values(by='share_count', ascending=False).reset_index()

                st.markdown("### 🔝 Top Coordinated Narratives")
                fig_nar = px.bar(
                    narrative_summary.head(10),
                    x='share_count',
                    y='shared_narrative',
                    orientation='h',
                    title="Top 10 Most Shared Narratives",
                    labels={'shared_narrative': 'Narrative Snippet', 'share_count': 'Share Count'},
                    color='share_count',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_nar, use_container_width=True)

                st.dataframe(narrative_summary)

                st.markdown("### 🔄 Full Similarity Pairs")
                st.dataframe(sim_df.drop(columns=['shared_narrative'], errors='ignore'))
            else:
                st.info("No significant textual similarities found above threshold.")
        except Exception as e:
            st.warning(f"⚠️ Similarity analysis failed: {e}")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")

    # Clustering
    try:
        clustered_df = cluster_texts(filtered_df)  # Should add 'cluster' column
        if 'cluster' not in clustered_df.columns:
            raise ValueError("Clustering did not return 'cluster' column")

        cluster_counts = clustered_df['cluster'].value_counts()

        st.markdown("### 🤖 Detected Coordination Clusters")
        fig_clust = px.bar(
            cluster_counts,
            title="Cluster Sizes (Potential Botnets/Campaigns)",
            labels={'value': 'Member Count', 'index': 'Cluster ID'},
            color=cluster_counts.index.astype(str),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_clust, use_container_width=True)

        st.dataframe(clustered_df[['Source', 'text', 'Timestamp', 'cluster']])
    except Exception as e:
        st.warning(f"⚠️ Clustering failed: {e}")

    # Network Graph
    st.markdown("### 🕸️ User Interaction Network")
    try:
        G, pos, cluster_map = build_user_interaction_graph(clustered_df if 'clustered_df' in locals() else filtered_df)

        # Create interactive Plotly network
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'))

        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=list(G.nodes()),
            mode='markers+text',
            textposition="top center",
            marker=dict(
                size=12,
                color=[cluster_map.get(node, 0) for node in G.nodes()],
                colorscale='Set3',
                colorbar=dict(title="Clusters"),
                line=dict(width=2, color='darkblue')
            ),
            hoverinfo='text'
        )

        fig_net = go.Figure(data=edge_trace + [node_trace],
                            layout=go.Layout(
                                title="User Network (Click & Drag to Explore)",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=60),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=600))
        st.plotly_chart(fig_net, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Network graph failed: {e}")

    # High-risk accounts
    st.markdown("### ⚠️ High-Risk Accounts (Spreading Multiple Narratives)")
    try:
        # Use sim_df from Tab 2
        if 'sim_df' in locals() and not sim_df.empty:
            all_sources = pd.concat([
                sim_df[['source1']].rename(columns={'source1': 'Source'}),
                sim_df[['source2']].rename(columns={'source2': 'Source'})
            ])['Source']
            source_counts = all_sources.value_counts()
            high_risk = source_counts[source_counts >= 3]  # At least 3 coordinated shares

            if len(high_risk) > 0:
                fig_hr = px.bar(
                    high_risk,
                    title="Accounts Involved in ≥3 Coordinated Messages",
                    labels={'value': 'Coordination Instances', 'index': 'Account'},
                    color='value',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_hr, use_container_width=True)
            else:
                st.info("No account found sharing 3+ coordinated messages.")
        else:
            st.info("No coordinated narratives detected to assess risk.")
    except Exception as e:
        st.warning(f"Could not compute risk scores: {e}")
