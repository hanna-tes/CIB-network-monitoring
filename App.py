import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
from io import StringIO

# Custom modules
try:
    from modules.embedding_utils import compute_text_similarity
except:
    # Fallback if not available, use TF-IDF directly
    def compute_text_similarity(text1, text2):
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
        vectorizer = TfidfVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2])
        return cosine_similarity(vectors)[0][1]

try:
    from modules.clustering_utils import cluster_texts, build_user_interaction_graph
except:
    # Minimal fallbacks
    def cluster_texts(df):
        df = df.copy()
        df['cluster'] = 0
        return df

    def build_user_interaction_graph(df):
        G = nx.Graph()
        sources = df['Source'].dropna().unique()[:10]  # Sample nodes
        for u in sources:
            G.add_node(u)
            for v in sources:
                if u != v:
                    G.add_edge(u, v, weight=1)
        pos = nx.spring_layout(G, seed=42)
        cluster_map = {n: 0 for n in G.nodes}
        return G, pos, cluster_map


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
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load default dataset: {e}")
        return pd.DataFrame()


def find_textual_similarities(df, threshold=0.85):
    """
    Fast vectorized version of similarity detection.
    Uses TF-IDF + cosine similarity matrix on 'original_text'.
    """
    clean_df = df[['original_text', 'Source', 'Timestamp']].dropna()
    clean_df = clean_df[clean_df['original_text'].str.strip() != ""]
    texts = clean_df['original_text'].tolist()

    if len(texts) < 2:
        return pd.DataFrame()

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    sim_matrix = np.triu(sim_matrix, k=1)  # Upper triangle only

    # Find high-similarity pairs
    idx_i, idx_j = np.where(sim_matrix >= threshold)
    seen = set()
    similar_pairs = []

    for i, j in zip(idx_i, idx_j):
        key = tuple(sorted([i, j]))
        if key in seen:
            continue
        seen.add(key)

        row1 = clean_df.iloc[i]
        row2 = clean_df.iloc[j]

        similar_pairs.append({
            'text1': row1['original_text'],
            'source1': row1['Source'],
            'time1': row1['Timestamp'],
            'text2': row2['original_text'],
            'source2': row2['Source'],
            'time2': row2['Timestamp'],
            'similarity': round(sim_matrix[i, j], 3),
            'shared_narrative': row1['original_text'][:150] + ("..." if len(row1['original_text']) > 150 else "")
        })

    return pd.DataFrame(similar_pairs)


# --- Cached Expensive Functions ---
@st.cache_data(show_spinner="🔍 Computing textual similarities...")
def cached_similarity_analysis(_df, threshold=0.85):
    return find_textual_similarities(_df, threshold)


@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df):
    return cluster_texts(_df)


@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df):
    G, pos, cluster_map = build_user_interaction_graph(_df)
    return G, pos, cluster_map


# --- Sidebar: Upload & Filters ---
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Load data
if uploaded_file:
    try:
        raw_bytes = uploaded_file.getvalue()
        encodings = ['utf-16', 'utf-8', 'latin1']
        decoded = None
        for enc in encodings:
            try:
                decoded = raw_bytes.decode(enc)
                st.sidebar.info(f"✅ Decoded using: {enc}")
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            st.error("Could not decode file.")
            st.stop()

        sample_line = decoded.strip().splitlines()[0]
        sep = '\t' if '\t' in sample_line else ','

        df = pd.read_csv(StringIO(decoded), sep=sep, low_memory=False)
    except Exception as e:
        st.sidebar.error(f"Failed to parse file: {e}")
        st.stop()
else:
    st.sidebar.info("Using default demo dataset")
    df = load_default_dataset()

# Exit if no data
if df is None or df.empty:
    st.warning("No data available. Please upload a valid dataset.")
    st.stop()


# --- Standardize Columns Safely ---
df.columns = [str(col).strip() for col in df.columns]

col_map = {
    # Twitter / Meltwater
    'Influencer': 'Source',
    'author': 'Source',
    'username': 'Source',
    'user': 'Source',
    'authorMeta/name': 'Source',
    'Hit Sentence': 'text',
    'Headline': 'text',
    'message': 'text',
    'title': 'text',
    'Date': 'Timestamp',
    'createTimeISO': 'Timestamp',

    # Telegram / Media
    'media_name': 'Outlet',           # Don't overwrite Source
    'channeltitle': 'Channel',        # Also not Source
    'published_date': 'Timestamp',
    'pubDate': 'Timestamp',

    # Generic video/text content
    'description': 'text',
    'content': 'text',
}


new_columns = []
for col in df.columns:
    if col in col_map:
        new_columns.append(col_map[col])
        continue
    normalized_col = col.lower().replace(" ", "").replace("_", "").replace("-", "")
    matched = None
    for key, target in col_map.items():
        norm_key = key.lower().replace(" ", "").replace("_", "").replace("-", "")
        if normalized_col == norm_key:
            matched = target
            break
    new_columns.append(matched if matched else col)
df.columns = new_columns
df = df.loc[:, ~df.columns.duplicated()]

# --- Ensure Required Columns Exist ---
required_cols = ["Source", "Timestamp", "text"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    suggestion_guide = {
        'Source': ['influencer', 'author', 'user', 'username', 'name'],
        'Timestamp': ['date', 'time', 'created', 'published'],
        'text': ['hit sentence', 'headline', 'opening text', 'message', 'content']
    }
    for col in missing_cols:
        close_matches = [c for c in df.columns if any(sugg in c.lower().replace(" ", "") for sugg in suggestion_guide.get(col, []))]
        if close_matches:
            st.info(f"💡 Did you mean to map `{close_matches[0]}` → `{col}`?")
        else:
            if col == "Source":
                df['Source'] = "Unknown"
                st.warning("⚠️ Using 'Unknown' for Source.")
            elif col == "text":
                st.error("🚫 No text column found. Cannot proceed.")
                st.stop()
            elif col == "Timestamp":
                df['Timestamp'] = pd.Timestamp.now()
                st.warning("⚠️ Using current time for Timestamp.")

# Final check
for col in required_cols:
    if col not in df.columns:
        st.error(f"Still missing: {col}")
        st.stop()

df = df.dropna(subset=['text']).reset_index(drop=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp']).reset_index(drop=True)

# Add Platform
if 'URL' in df.columns and 'Platform' not in df.columns:
    df['Platform'] = df['URL'].apply(infer_platform_from_url)
elif 'Platform' not in df.columns:
    df['Platform'] = "Unknown"

# Extract original text
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
    start_dt = pd.Timestamp(date_range[0])
    end_dt = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
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

    top_sources = filtered_df['Source'].value_counts().head(10)
    fig_src = px.bar(top_sources, title="Top 10 Active Sources", labels={'value': 'Posts', 'index': 'Source'})
    st.plotly_chart(fig_src, use_container_width=True)

    filtered_df['hashtags'] = filtered_df['text'].str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
    all_hashtags = [tag for tags in filtered_df['hashtags'] for tag in tags]
    if all_hashtags:
        hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
        fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags", labels={'value': 'Frequency', 'index': 'Hashtag'})
        st.plotly_chart(fig_ht, use_container_width=True)
    else:
        st.info("No hashtags found.")

    time_series = filtered_df.set_index('Timestamp').resample('D').size()
    fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'Timestamp': 'Date'})
    st.plotly_chart(fig_ts, use_container_width=True)


# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🧠 Narrative Detection & Coordination")

    # Limit analysis size
    MAX_ROWS = st.sidebar.slider("Max posts to analyze", 100, 1000, 300)
    analysis_df = filtered_df.head(MAX_ROWS).copy()

    with st.spinner(f"🔍 Finding coordinated narratives among {len(analysis_df)} posts..."):
        sim_df = cached_similarity_analysis(analysis_df, threshold=0.85)

    if not sim_df.empty:
        st.success(f"✅ Found {len(sim_df)} similar pairs.")

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
        st.info("No significant similarities found above threshold.")


# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")

    try:
        clustered_df = cached_clustering(filtered_df)
        cluster_counts = clustered_df['cluster'].value_counts()

        st.markdown("### 🤖 Detected Coordination Clusters")
        fig_clust = px.bar(
            cluster_counts,
            title="Cluster Sizes",
            labels={'value': 'Member Count', 'index': 'Cluster ID'},
            color=cluster_counts.index.astype(str),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_clust, use_container_width=True)

        st.dataframe(clustered_df[['Source', 'text', 'Timestamp', 'cluster']])
    except Exception as e:
        st.warning(f"⚠️ Clustering failed: {e}")

    st.markdown("### 🕸️ User Interaction Network")
    try:
        G, pos, cluster_map = cached_network_graph(clustered_df if 'clustered_df' in locals() else filtered_df)
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

    st.markdown("### ⚠️ High-Risk Accounts")
    try:
        if 'sim_df' in locals() and not sim_df.empty:
            all_sources = pd.concat([
                sim_df[['source1']].rename(columns={'source1': 'Source'}),
                sim_df[['source2']].rename(columns={'source2': 'Source'})
            ])['Source']
            source_counts = all_sources.value_counts()
            high_risk = source_counts[source_counts >= 3]
            if len(high_risk) > 0:
                fig_hr = px.bar(
                    high_risk,
                    title="Accounts in ≥3 Coordinated Messages",
                    labels={'value': 'Coordination Instances', 'index': 'Account'},
                    color='value',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_hr, use_container_width=True)
            else:
                st.info("No account shared 3+ narratives.")
        else:
            st.info("No coordinated narratives detected.")
    except Exception as e:
        st.warning(f"Risk analysis failed: {e}")
