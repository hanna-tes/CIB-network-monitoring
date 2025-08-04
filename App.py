import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from itertools import combinations
import re
from io import BytesIO

# --- Set Page Config ---
st.set_page_config(page_title="CIB Dashboard", layout="wide")
st.title("🕵️ CIB Network Monitoring Dashboard")

# --- Helper Functions ---
def infer_platform_from_url(url):
    """Infers the social media or news platform from a given URL."""
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    if "tiktok.com" in url:
        return "TikTok"
    elif "facebook.com" in url or "fb.watch" in url:
        return "Facebook"
    elif "twitter.com" in url or "x.com" in url:
        return "X"
    elif "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    elif "instagram.com" in url:
        return "Instagram"
    elif "telegram.me" in url or "t.me" in url:
        return "Telegram"
    elif url.startswith("https://") or url.startswith("http://"):
        media_domains = ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "aljazeera.com", "lemonde.fr", "dw.com"]
        if any(domain in url for domain in media_domains):
            return "News/Media"
        return "Media"
    else:
        return "Unknown"

def extract_original_text(text):
    """
    Cleans text by removing RT/QT prefixes, @mentions, URLs, and normalizing spaces.
    Used for similarity analysis.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

# --- Robust Timestamp Parser: Returns UNIX Timestamp (Integer) ---
def parse_timestamp_robust(timestamp):
    """
    Converts a timestamp string to a UNIX timestamp (integer seconds since epoch).
    Returns None if parsing fails.
    """
    if pd.isna(timestamp):
        return None
    if isinstance(timestamp, (int, float)):
        if 0 < timestamp < 253402300800:  # Valid range: 1970–9999
            return int(timestamp)
        else:
            return None

    # List of common timestamp formats
    date_formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p',
        '%A, %d %b %Y %H:%M:%S', '%b %d, %I:%M%p', '%d %b %Y %I:%M%p',
        '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y',
    ]

    # Try direct parsing
    try:
        parsed = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.notna(parsed):
            return int(parsed.timestamp())
    except:
        pass

    # Try each format
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed):
                return int(parsed.timestamp())
        except (ValueError, TypeError):
            continue
    return None

# --- Combine Multiple Datasets with Flexible Object Column ---
def combine_social_media_data(
    meltwater_df,
    civicsignals_df,
    openmeasure_df=None,
    meltwater_object_col='hit sentence',
    civicsignals_object_col='title',
    openmeasure_object_col='text'
):
    """
    Combines datasets from Meltwater, CivicSignals, and Open-Measure (optional).
    Allows specification of which column to use as 'object_id' for coordination analysis.
    Returns timestamp as UNIX integer.
    """
    combined_dfs = []

    def get_specific_col(df, col_name_lower):
        if col_name_lower in df.columns:
            return df[col_name_lower]
        return pd.Series([np.nan] * len(df), index=df.index)

    # Process Meltwater
    if meltwater_df is not None and not meltwater_df.empty:
        meltwater_df.columns = meltwater_df.columns.str.lower()
        mw = pd.DataFrame()
        mw['account_id'] = get_specific_col(meltwater_df, 'influencer')
        mw['content_id'] = get_specific_col(meltwater_df, 'tweet id')
        mw['object_id'] = get_specific_col(meltwater_df, meltwater_object_col.lower())
        mw['original_url'] = get_specific_col(meltwater_df, 'url')
        mw['timestamp_share'] = get_specific_col(meltwater_df, 'date')
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)

    # Process CivicSignals
    if civicsignals_df is not None and not civicsignals_df.empty:
        civicsignals_df.columns = civicsignals_df.columns.str.lower()
        cs = pd.DataFrame()
        cs['account_id'] = get_specific_col(civicsignals_df, 'media_name')
        cs['content_id'] = get_specific_col(civicsignals_df, 'stories_id')
        cs['object_id'] = get_specific_col(civicsignals_df, civicsignals_object_col.lower())
        cs['original_url'] = get_specific_col(civicsignals_df, 'url')
        cs['timestamp_share'] = get_specific_col(civicsignals_df, 'publish_date')
        cs['source_dataset'] = 'CivicSignals'
        combined_dfs.append(cs)

    # Process Open-Measure
    if openmeasure_df is not None and not openmeasure_df.empty:
        openmeasure_df.columns = openmeasure_df.columns.str.lower()
        om = pd.DataFrame()
        om['account_id'] = get_specific_col(openmeasure_df, 'actor_username')
        om['content_id'] = get_specific_col(openmeasure_df, 'id')
        om['object_id'] = get_specific_col(openmeasure_df, openmeasure_object_col.lower())
        om['original_url'] = get_specific_col(openmeasure_df, 'url')
        om['timestamp_share'] = get_specific_col(openmeasure_df, 'created_at')
        om['source_dataset'] = 'OpenMeasure'
        combined_dfs.append(om)

    if not combined_dfs:
        return pd.DataFrame()

    combined = pd.concat(combined_dfs, ignore_index=True)
    combined = combined.dropna(subset=['account_id', 'content_id', 'timestamp_share', 'object_id']).copy()
    combined['account_id'] = combined['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    combined['content_id'] = combined['content_id'].astype(str).str.replace('"', '', regex=False).str.strip()
    combined['original_url'] = combined['original_url'].astype(str).replace('nan', '').fillna('')
    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')

    # Convert timestamp to UNIX
    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)
    combined['timestamp_share'] = combined['timestamp_share'].astype('Int64')  # Nullable integer

    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')
    combined = combined[combined['object_id'].str.strip() != ""].copy()
    combined = combined.drop_duplicates(subset=['account_id', 'content_id', 'object_id', 'timestamp_share']).reset_index(drop=True)
    return combined

# --- Final Preprocessing Function ---
def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    """
    Performs final preprocessing steps on the combined DataFrame.
    Respects coordination_mode: uses text or URL as object_id.
    Ensures timestamp_share is UNIX integer.
    """
    if df.empty:
        return df
    df_processed = df.copy()
    df_processed.rename(columns={'original_url': 'URL'}, inplace=True)
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].copy()

    def clean_text_for_display(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    if coordination_mode == "Text Content":
        df_processed['object_id'] = df_processed['object_id'].apply(clean_text_for_display)
        df_processed = df_processed[df_processed['object_id'].str.len() > 0].reset_index(drop=True)
        df_processed = df_processed[
            ~df_processed['object_id'].str.lower().str.startswith('rt @') &
            ~df_processed['object_id'].str.lower().str.startswith('qt @')
        ].reset_index(drop=True)

    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    elif coordination_mode == "Shared URLs":
        df_processed['original_text'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')

    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)

    if 'Outlet' not in df_processed.columns:
        df_processed['Outlet'] = np.nan
    if 'Channel' not in df_processed.columns:
        df_processed['Channel'] = np.nan

    if df_processed.empty:
        st.error("❌ No valid data after final preprocessing.")
        st.stop()

    return df_processed

# --- Analysis Functions ---
def cluster_texts(df, eps=0.3, min_samples=2):
    if 'original_text' not in df.columns:
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy
    texts_to_cluster = df['original_text'].astype(str).tolist()
    if not texts_to_cluster or all(t.strip() == "" for t in texts_to_cluster):
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts_to_cluster)
    except ValueError as e:
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy
    eps = max(0.01, min(0.99, eps))
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(tfidf_matrix)
    df_copy = df.copy()
    df_copy['cluster'] = clustering.labels_
    return df_copy

def build_user_interaction_graph(df, coordination_type="text"):
    G = nx.Graph()
    influencer_column = 'account_id'

    if coordination_type == "text":
        if 'cluster' not in df.columns:
            return G, {}, {}
        grouped = df.groupby('cluster')
        for cluster_id, group in grouped:
            if cluster_id == -1 or len(group[influencer_column].unique()) < 2:
                for user in group[influencer_column].dropna().unique():
                    if user not in G:
                        G.add_node(user, cluster=cluster_id)
                continue
            users_in_cluster = group[influencer_column].dropna().unique().tolist()
            for u1, u2 in combinations(users_in_cluster, 2):
                if G.has_edge(u1, u2):
                    G[u1][u2]['weight'] += 1
                else:
                    G.add_edge(u1, u2, weight=1)

    elif coordination_type == "url":
        if 'URL' not in df.columns:
            return G, {}, {}
        url_groups = df.groupby('URL')
        for url_shared, group in url_groups:
            if pd.isna(url_shared) or url_shared.strip() == "":
                continue
            users_sharing_url = group[influencer_column].dropna().unique().tolist()
            if len(users_sharing_url) < 2:
                for user in users_sharing_url:
                    if user not in G:
                        G.add_node(user)
                continue
            for u1, u2 in combinations(users_sharing_url, 2):
                if G.has_edge(u1, u2):
                    G[u1][u2]['weight'] += 1
                else:
                    G.add_edge(u1, u2, weight=1)

    all_influencers = df[influencer_column].dropna().unique().tolist()
    influencer_platform_map = df.groupby(influencer_column)['Platform'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').to_dict()

    for inf in all_influencers:
        if inf not in G.nodes():
            G.add_node(inf)
        G.nodes[inf]['platform'] = influencer_platform_map.get(inf, 'Unknown')
        if coordination_type == "text":
            clusters = df[df[influencer_column] == inf]['cluster'].dropna()
            G.nodes[inf]['cluster'] = clusters.mode()[0] if not clusters.empty else -2
        elif coordination_type == "url":
            shared_urls = df[(df[influencer_column] == inf) & df['URL'].notna() & (df['URL'].str.strip() != '')]['URL'].unique()
            G.nodes[inf]['cluster'] = f"SharedURL_Group_{hash(tuple(sorted(shared_urls))) % 100}" if len(shared_urls) > 0 else "NoSharedURL"

    pos = nx.spring_layout(G, seed=42, k=0.1, iterations=50)
    cluster_map = {node: G.nodes[node].get('cluster', -2) for node in G.nodes()}
    return G, pos, cluster_map

def find_textual_similarities(df, threshold=0.85):
    clean_df = df[['original_text', 'account_id', 'timestamp_share', 'Platform', 'URL']].copy()
    clean_df['original_text'] = clean_df['original_text'].astype(str)
    clean_df = clean_df.dropna(subset=['original_text', 'account_id', 'timestamp_share'])
    clean_df = clean_df[clean_df['original_text'].str.strip() != ""].copy()
    texts = clean_df['original_text'].tolist()
    if len(texts) < 2:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as e:
        st.warning(f"TF-IDF failed: {e}")
        return pd.DataFrame()
    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    sim_matrix = np.triu(sim_matrix, k=1)
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
        snippet = row1['original_text'][:150] + ("..." if len(row1['original_text']) > 150 else "")
        if not snippet.strip():
            snippet = "Empty/Cleaned Text"
        similar_pairs.append({
            'text1': row1['original_text'],
            'account_id1': row1['account_id'],
            'platform1': row1['Platform'],
            'timestamp_share1': row1['timestamp_share'],
            'url1': row1['URL'],
            'text2': row2['original_text'],
            'account_id2': row2['account_id'],
            'platform2': row2['Platform'],
            'timestamp_share2': row2['timestamp_share'],
            'url2': row2['URL'],
            'similarity': round(sim_matrix[i, j], 3),
            'shared_narrative': snippet,
            'platforms_involved': f"{row1['Platform']},{row2['Platform']}"
        })
    return pd.DataFrame(similar_pairs)

# --- Cached Functions ---
@st.cache_data(show_spinner="🔍 Computing textual similarities...")
def cached_similarity_analysis(_df, threshold=0.85, data_source="default"):
    return find_textual_similarities(_df, threshold)

@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df, data_source="default"):
    return cluster_texts(_df)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df_for_graph, coordination_type="text", data_source="default"):
    return build_user_interaction_graph(_df_for_graph, coordination_type)

# --- Sidebar: Data Source & Coordination Mode ---
st.sidebar.header("📥 Data Source")
data_source = st.sidebar.radio("Choose data source:", ("Use Default Datasets", "Upload CSV Files"))

# Coordination Mode Selector
st.sidebar.header("🎯 Coordination Analysis Mode")
coordination_mode = st.sidebar.radio(
    "Analyze coordination by:",
    ("Text Content", "Shared URLs"),
    help="Choose what defines a coordinated action: similar text messages or sharing the same external link."
)

# Clear cache when mode or source changes
if 'last_data_source' not in st.session_state or st.session_state.last_data_source != data_source:
    st.cache_data.clear()
    st.session_state.last_data_source = data_source
if 'last_coordination_mode' not in st.session_state or st.session_state.last_coordination_mode != coordination_mode:
    st.cache_data.clear()
st.session_state.last_coordination_mode = coordination_mode

combined_raw_df = pd.DataFrame()

# Load data
if data_source == "Use Default Datasets":
    st.sidebar.info("Using default datasets from GitHub.")
    with st.spinner("📥 Loading and combining default datasets..."):
        base_url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/"
        urls = {
            "meltwater": f"{base_url}TogoJULYData%20-%20Sheet1.csv",
            "civicsignals": f"{base_url}togo-or-lome-or-togo-all-story-urls-20250707142808.csv"
        }
        meltwater_df = pd.DataFrame()
        civicsignals_df = pd.DataFrame()
        for key, url in urls.items():
            try:
                df = pd.read_csv(url, sep=',')
                if not df.empty:
                    if key == "meltwater":
                        meltwater_df = df
                    elif key == "civicsignals":
                        civicsignals_df = df
                    st.sidebar.success(f"✅ {key.capitalize()}: Loaded {len(df)} rows")
                else:
                    st.sidebar.warning(f"⚠️ {key.capitalize()}: Empty file.")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Failed to load {key}: {e}")

        obj_map = {
            "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
            "civicsignals": "title" if coordination_mode == "Text Content" else "url",
            "openmeasure": "text" if coordination_mode == "Text Content" else "url"
        }
        combined_raw_df = combine_social_media_data(
            meltwater_df if not meltwater_df.empty else None,
            civicsignals_df if not civicsignals_df.empty else None,
            None,
            meltwater_object_col=obj_map["meltwater"],
            civicsignals_object_col=obj_map["civicsignals"],
            openmeasure_object_col=obj_map["openmeasure"]
        )
    if combined_raw_df.empty:
        st.warning("No data loaded from default datasets.")
        st.stop()
    st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from default datasets.")

elif data_source == "Upload CSV Files":
    st.sidebar.info("Upload your CSV files below.")
    uploaded_meltwater = st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater_upload")
    uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
    uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")

    meltwater_df_upload = pd.DataFrame()
    civicsignals_df_upload = pd.DataFrame()
    openmeasure_df_upload = pd.DataFrame()

    if uploaded_meltwater:
        bytes_data = uploaded_meltwater.getvalue()
        try:
            meltwater_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',', low_memory=False)
            st.sidebar.success(f"✅ Meltwater: Loaded {len(meltwater_df_upload)} rows")
        except Exception as e:
            st.error(f"❌ Failed to read Meltwater CSV: {e}")
            st.stop()

    if uploaded_civicsignals:
        bytes_data = uploaded_civicsignals.getvalue()
        try:
            civicsignals_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',')
            st.sidebar.success(f"✅ CivicSignal: Loaded {len(civicsignals_df_upload)} rows")
        except Exception as e:
            st.error(f"❌ Failed to read CivicSignals CSV: {e}")
            st.stop()

    if uploaded_openmeasure:
        bytes_data = uploaded_openmeasure.getvalue()
        try:
            openmeasure_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',', low_memory=False)
            st.sidebar.success(f"✅ Open-Measure: Loaded {len(openmeasure_df_upload)} rows")
        except Exception as e:
            st.error(f"❌ Failed to read Open-Measure CSV: {e}")
            st.stop()

    if not meltwater_df_upload.empty or not civicsignals_df_upload.empty or not openmeasure_df_upload.empty:
        obj_map = {
            "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
            "civicsignals": "title" if coordination_mode == "Text Content" else "url",
            "openmeasure": "text" if coordination_mode == "Text Content" else "url"
        }
        with st.spinner("Combining uploaded datasets..."):
            combined_raw_df = combine_social_media_data(
                meltwater_df_upload if not meltwater_df_upload.empty else None,
                civicsignals_df_upload if not civicsignals_df_upload.empty else None,
                openmeasure_df_upload if not openmeasure_df_upload.empty else None,
                meltwater_object_col=obj_map["meltwater"],
                civicsignals_object_col=obj_map["civicsignals"],
                openmeasure_object_col=obj_map["openmeasure"]
            )
        st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from uploaded files.")
    else:
        st.warning("Please upload at least one CSV file to proceed.")
        st.stop()

# Debug
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Mode:** `{coordination_mode}`")
st.sidebar.markdown(f"**Source:** `{data_source}`")
st.sidebar.markdown(f"**Total Rows After Combine:** `{len(combined_raw_df):,}`")

# --- Final Preprocess ---
with st.spinner("⏳ Preprocessing and mapping combined data..."):
    df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode=coordination_mode)

if df.empty:
    st.error("❌ No valid data after final preprocessing.")
    st.stop()

# --- Download Combined Data ---
st.sidebar.markdown("### 💾 Download Combined & Preprocessed Data")
@st.cache_data
def convert_df_to_csv(data_frame):
    return data_frame.to_csv(index=False).encode('utf-8')

download_df_columns = ['account_id', 'content_id', 'object_id', 'timestamp_share']
downloadable_df = df[download_df_columns].copy() if all(col in df.columns for col in download_df_columns) else pd.DataFrame()

if not downloadable_df.empty:
    combined_preprocessed_csv = convert_df_to_csv(downloadable_df)
    st.sidebar.download_button(
        "Download Preprocessed Dataset (Core Columns)",
        combined_preprocessed_csv,
        f"preprocessed_combined_core_data_{coordination_mode.replace(' ', '_').lower()}.csv",
        "text/csv",
        help="Downloads the data after all preprocessing and column mapping. 'object_id' contains either text or URL based on your selection."
    )
else:
    st.sidebar.warning("Could not create downloadable dataset with core columns.")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters (Apply to all tabs)")
if 'timestamp_share' not in df.columns or df['timestamp_share'].dtype != 'Int64':
    st.error("timestamp_share must be an integer (UNIX timestamp).")
    st.stop()

min_date = pd.to_datetime(df['timestamp_share'].min(), unit='s').date()
max_date = pd.to_datetime(df['timestamp_share'].max(), unit='s').date()
selected_date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

if len(selected_date_range) == 2:
    start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
    end_ts = int((pd.Timestamp(selected_date_range[1], tz='UTC') + timedelta(days=1) - timedelta(microseconds=1)).timestamp())
else:
    start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
    end_ts = start_ts + 86400 - 1

filtered_df_global = df[
    (df['timestamp_share'] >= start_ts) &
    (df['timestamp_share'] <= end_ts) &
    (df['Platform'].isin(df['Platform'].dropna().unique()))
].copy()

if filtered_df_global.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# Export filtered data
st.sidebar.markdown("### 📄 Export Filtered Results")
filtered_csv_data = convert_df_to_csv(filtered_df_global)
st.sidebar.download_button("Download Filtered Data (All Columns)", filtered_csv_data, "filtered_dashboard_data.csv", "text/csv")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Summary Statistics")
    st.markdown("### 🔬 Preprocessed Data Sample")
    st.markdown(f"**Data Source:** `{data_source}` | **Coordination Mode:** `{coordination_mode}` | **Total Rows:** `{len(df):,}`")
    display_cols_overview = ['account_id', 'content_id', 'object_id', 'timestamp_share']
    existing_cols = [col for col in df.columns if col in display_cols_overview]
    st.dataframe(df[existing_cols].head(10))

    if 'source_dataset' in filtered_df_global.columns:
        st.markdown("### 📊 Data Sources in Filtered Data")
        source_counts = filtered_df_global['source_dataset'].value_counts()
        st.dataframe(source_counts)

    if not filtered_df_global.empty:
        top_influencers = filtered_df_global['account_id'].value_counts().head(10)
        fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'account_id'})
        st.plotly_chart(fig_src, use_container_width=True)
        st.markdown("**Top 10 Influencers**: Shows the most active accounts based on number of posts.")

        if 'Platform' in filtered_df_global.columns:
            all_platforms_counts = filtered_df_global['Platform'].value_counts()
            fig_platform = px.bar(all_platforms_counts, title="Post Distribution by Platform", labels={'value': 'Posts', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True)
            st.markdown("**Post Distribution by Platform**: Visualizes how posts are distributed across different social and media platforms.")

        if 'Outlet' in filtered_df_global.columns and filtered_df_global['Outlet'].notna().any():
            top_outlets = filtered_df_global['Outlet'].value_counts().head(10)
            fig_outlet = px.bar(top_outlets, title="Top 10 Media Outlets/Channels", labels={'value': 'Posts', 'index': 'Outlet'})
            st.plotly_chart(fig_outlet, use_container_width=True)
            st.markdown("**Top 10 Media Outlets/Channels**: Ranks traditional and digital media sources by volume of coverage.")
        elif 'Channel' in filtered_df_global.columns and filtered_df_global['Channel'].notna().any():
            top_channels = filtered_df_global['Channel'].value_counts().head(10)
            fig_chan = px.bar(top_channels, title="Top 10 Channels", labels={'value': 'Posts', 'index': 'Channel'})
            st.plotly_chart(fig_chan, use_container_width=True)
            st.markdown("**Top 10 Channels**: Displays the most active YouTube or social media channels.")

        social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media'])].copy()
        if not social_media_df.empty and 'object_id' in social_media_df.columns:
            social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
            all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
            if all_hashtags:
                hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                st.plotly_chart(fig_ht, use_container_width=True)
                st.markdown("**Top 10 Hashtags (Social Media Only)**: Highlights the most frequently used hashtags on social platforms.")

        # ✅ Fixed: Safe UNIX timestamp conversion for plotting
        plot_df = filtered_df_global.copy()

        if 'timestamp_share' not in plot_df.columns:
            st.warning("⚠️ 'timestamp_share' column not found. Cannot plot time series.")
        else:
            # Convert to numeric (in case it's string)
            plot_df['timestamp_share'] = pd.to_numeric(plot_df['timestamp_share'], errors='coerce')
            # Keep only valid UNIX timestamps (roughly between 2000–2100)
            valid_mask = (plot_df['timestamp_share'] >= 946684800) & (plot_df['timestamp_share'] <= 4102444800)
            plot_df = plot_df[valid_mask]
            if plot_df.empty:
                st.info("No valid timestamps available for time series.")
            else:
                # Convert to datetime
                plot_df['datetime'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True)
                plot_df = plot_df.set_index('datetime')
                time_series = plot_df.resample('D').size()

                if time_series.empty:
                    st.info("No data to display in time series.")
                else:
                    fig_ts = px.area(
                        time_series,
                        title="Daily Post Volume",
                        labels={'value': 'Number of Posts', 'datetime': 'Date'},
                        markers=True
                    )
                    fig_ts.update_layout(xaxis_title="Date", yaxis_title="Number of Posts")
                    st.plotly_chart(fig_ts, use_container_width=True)
                    st.markdown("**Daily Post Volume**: Visualizes the volume of posts over time to identify spikes or trends.")
                    
# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🧠 Narrative Detection & Coordination")
    st.markdown(f"**Current Mode:** Analyzing coordination by **{coordination_mode}**")
    st.markdown("---")

    if coordination_mode == "Text Content":
        platforms_analysis = st.multiselect(
            "Platforms to include in Similarity Analysis:",
            options=filtered_df_global['Platform'].dropna().astype(str).unique().tolist(),
            default=filtered_df_global['Platform'].dropna().astype(str).unique().tolist(),
            key="platforms_analysis_tab2"
        )

        MAX_ROWS_SIMILARITY = st.slider(
            "Max posts to analyze for similarity (for performance)",
            100, 1000, 300,
            key="max_rows_similarity"
        )

        # Filter by platform
        analysis_df_filtered_by_platform = filtered_df_global[
            filtered_df_global['Platform'].isin(platforms_analysis)
        ].copy()

        # 🔥 Keep only original tweets (no RT, QT, repost)
        original_df = analysis_df_filtered_by_platform[
            ~analysis_df_filtered_by_platform['object_id'].astype(str).str.startswith('RT @') &
            ~analysis_df_filtered_by_platform['object_id'].astype(str).str.startswith('qt @') &
            ~analysis_df_filtered_by_platform['object_id'].astype(str).str.contains('repost', case=False, na=False)
        ].copy()

        # Ensure original_text is available
        if 'original_text' not in original_df.columns:
            original_df['original_text'] = original_df['object_id'].astype(str)

        # Clean and limit
        analysis_df = original_df[
            (original_df['original_text'].astype(str).str.strip() != "") &
            (original_df['original_text'].str.lower() != "nan")
        ].head(MAX_ROWS_SIMILARITY).copy()

        if len(analysis_df) < 2:
            st.info("Not enough original posts for similarity analysis.")
        else:
            st.markdown("### 🔍 Text Similarity Analysis on Original Posts Only")
            st.caption("Only original tweets (non-RT, non-QT, non-repost) are analyzed to detect true narrative coordination.")

            with st.spinner(f"🔍 Finding coordinated narratives among {len(analysis_df)} original posts..."):
                sim_df = cached_similarity_analysis(analysis_df, threshold=0.85)

            if not sim_df.empty:
                st.success(f"✅ Found {len(sim_df)} similar pairs among original posts.")

                # Add repost count for each narrative
                def get_repost_count(narrative):
                    pattern = re.escape(narrative.lower()[:20])
                    repost_df = filtered_df_global[
                        filtered_df_global['object_id'].astype(str).str.contains('repost', case=False, na=False) |
                        filtered_df_global['object_id'].astype(str).str.startswith('RT @') |
                        filtered_df_global['object_id'].astype(str).str.startswith('qt @')
                    ]
                    matches = repost_df['object_id'].astype(str).str.contains(pattern, case=False, na=False)
                    return matches.sum()

                # Aggregate narratives
                try:
                    narrative_summary = sim_df.groupby('shared_narrative').agg(
                        share_count=('similarity', 'count'),
                        influencers_involved=('account_id1', lambda x: ", ".join(x.astype(str).unique()[:5]) + ("..." if len(x.unique()) > 5 else "")),
                        platforms_involved=('platform1', lambda x: ", ".join(
                            sorted(
                                list(
                                    set(p.strip() for p in x.astype(str).tolist() if p.strip() != "")
                                )
                            )
                        )),
                        repost_count=('shared_narrative', lambda x: get_repost_count(x.iloc[0]))
                    ).sort_values(by='share_count', ascending=False).reset_index()

                    st.markdown("### 🔝 Top Coordinated Narratives")
                    fig_nar = px.bar(
                        narrative_summary.head(10),
                        x='share_count',
                        y='shared_narrative',
                        orientation='h',
                        title="Top 10 Most Shared Narratives",
                        labels={'shared_narrative': 'Narrative Snippet', 'share_count': 'Copy-Paste Count'},
                        color='repost_count',
                        color_continuous_scale='Blues',
                        hover_data=['repost_count']
                    )
                    st.plotly_chart(fig_nar, use_container_width=True)
                    st.markdown("**Top Coordinated Narratives**: Based on original posts. Bar color shows how many times the narrative was amplified via reposts.")

                    st.dataframe(narrative_summary)
                    st.markdown("### 🔄 Full Similarity Pairs")
                    st.caption("Only shows similarity between original posts (no RTs, QTs, or reposts). Use links to verify context.")

                    # Prepare display DataFrame
                    display_sim_df = sim_df[[
                        'text1', 'account_id1', 'platform1', 'timestamp_share1', 'url1',
                        'text2', 'account_id2', 'platform2', 'timestamp_share2', 'url2',
                        'similarity'
                    ]].copy()

                    # Rename for clarity
                    display_sim_df = display_sim_df.rename(columns={
                        'account_id1': 'Influencer 1',
                        'account_id2': 'Influencer 2',
                        'platform1': 'Platform 1',
                        'platform2': 'Platform 2',
                        'timestamp_share1': 'Time 1',
                        'timestamp_share2': 'Time 2',
                        'url1': 'URL 1',
                        'url2': 'URL 2',
                        'similarity': 'Similarity'
                    })

                    # Format timestamps
                    for col in ['Time 1', 'Time 2']:
                        if col in display_sim_df.columns:
                            display_sim_df[col] = pd.to_datetime(
                                display_sim_df[col], unit='s', utc=True, errors='coerce'
                            ).dt.strftime('%Y-%m-%d %H:%M')

                    # Use st.data_editor with LinkColumn
                    st.data_editor(
                        display_sim_df,
                        column_config={
                            "URL 1": st.column_config.LinkColumn("URL 1"),
                            "URL 2": st.column_config.LinkColumn("URL 2")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    st.markdown("**Full Similarity Pairs**: Shows all pairs of posts with highly similar content, including source and timestamps.")
                except Exception as e:
                    st.error(f"❌ Failed to generate narrative summary: {e}")
                    st.write("Debug: Check column names in `sim_df`:")
                    st.write(list(sim_df.columns))
            else:
                st.info("No significant similarities found above threshold between original posts.")
    else:
        st.subheader("🔗 Coordination by Shared URLs")
        url_df = filtered_df_global[
            (filtered_df_global['URL'].notna()) &
            (filtered_df_global['URL'].str.strip() != "") &
            (filtered_df_global['URL'] != "Unknown")
        ].copy()
        if not url_df.empty:
            url_counts = url_df.groupby('URL')['account_id'].nunique()
            shared_urls = url_counts[url_counts >= 2].index.tolist()
            if shared_urls:
                st.success(f"✅ Found {len(shared_urls)} URLs shared by multiple influencers.")
                url_summary = url_df[url_df['URL'].isin(shared_urls)].groupby('URL').agg(
                    share_count=('account_id', 'nunique'),
                    influencers=('account_id', lambda x: ", ".join(x.unique())),
                    platforms=('Platform', lambda x: ", ".join(sorted(x.unique())))
                ).sort_values(by='share_count', ascending=False).reset_index()
                st.data_editor(
                    url_summary,
                    column_config={"URL": st.column_config.LinkColumn("Shared URL")},
                    hide_index=True,
                    use_container_width=True
                )
                st.markdown("**Shared URLs Analysis**: Identifies content amplification through URL sharing across accounts.")
            else:
                st.info("No URLs shared by multiple influencers.")
        else:
            st.info("No valid URLs to analyze.")
            
# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")
    st.markdown("---")

    # ✅ Debug: Show columns
    #st.write("🔍 **Available columns:**", list(filtered_df_global.columns))

    risk_df = filtered_df_global.copy()

    # 🔁 Force 'text' column from possible sources
    text_cols = ['Hit Sentence', 'title', 'message', 'content', 'object_id', 'Body']
    for col in text_cols:
        if col in risk_df.columns:
            risk_df['text'] = risk_df[col].astype(str).replace('nan', '').str.strip()
            st.info(f"✅ Using '{col}' as 'text'")
            break
    else:
        st.error("❌ No text column found.")
        st.stop()

    # Clean
    risk_df = risk_df[risk_df['text'] != ""].reset_index(drop=True)
    if risk_df.empty:
        st.error("❌ No valid text after cleaning.")
        st.stop()

    # 🔁 Convert timestamp
    if 'timestamp_share' in risk_df.columns:
        risk_df['Timestamp'] = pd.to_datetime(risk_df['timestamp_share'], unit='s', utc=True, errors='coerce')
        risk_df = risk_df.dropna(subset=['Timestamp']).reset_index(drop=True)
    else:
        st.warning("⚠️ Using current time for Timestamp.")
        risk_df['Timestamp'] = pd.Timestamp.now(tz='UTC')

    if 'original_text' not in risk_df.columns:
        risk_df['original_text'] = risk_df['text'].apply(extract_original_text)

    # ----------------------------
    # 🤖 Detected Coordination Clusters
    # ----------------------------
    st.markdown("### 🤖 Detected Coordination Clusters")
    st.markdown("**Detected Coordination Clusters**: Groups posts into clusters based on content similarity, revealing coordinated campaigns.")
    try:
        clustered_df = cached_clustering(risk_df)
        if 'cluster' not in clustered_df.columns:
            raise ValueError("Clustering did not return 'cluster' column")

        cluster_counts = clustered_df['cluster'].value_counts()
        if cluster_counts.empty:
            st.info("No clusters detected.")
        else:
            fig_clust = px.bar(
                cluster_counts,
                title="Cluster Sizes",
                labels={'value': 'Member Count', 'index': 'Cluster ID'},
                color=cluster_counts.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_clust, use_container_width=True)
            st.dataframe(clustered_df[['account_id', 'text', 'Timestamp', 'cluster']])
    except Exception as e:
        st.warning(f"⚠️ Clustering failed: {e}")
        # Fallback: assign all to one cluster
        risk_df['cluster'] = 0
        clustered_df = risk_df
        st.info("Using all data in a single cluster for network analysis.")

    # ----------------------------
    # 🕸️ User Interaction Network (Limited)
    # ----------------------------
    st.markdown("### 🕸️ User Interaction Network")
    st.markdown("""
        **User Interaction Network**: Visualizes connections between the most active influencers who share similar content. 
        Only the top influencers are shown to ensure performance and clarity.
    """)

    try:
        # Use clustered_df if available, else fall back
        graph_input_df = clustered_df if 'clustered_df' in locals() and not clustered_df.empty else risk_df

        # 🔥 Limit to top influencers
        MAX_NODES = st.slider(
            "Max Influencers in Network Graph (for performance)",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            key="max_nodes_tab3"
        )

        # Get top influencers by post count
        top_influencers = graph_input_df['account_id'].value_counts().nlargest(MAX_NODES).index.tolist()
        graph_subset = graph_input_df[graph_input_df['account_id'].isin(top_influencers)].copy()

        if graph_subset.empty:
            st.info("No influencers to display in the network.")
        else:
            # Ensure original_text
            if 'original_text' not in graph_subset.columns:
                graph_subset['original_text'] = graph_subset['text'].apply(extract_original_text)

            # Build graph
            G, pos, cluster_map = cached_network_graph(graph_subset)

            if G is None or len(G.nodes()) == 0:
                st.info("No nodes to display in the network graph.")
            else:
                edge_trace = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.append(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'
                    ))

                node_colors = [cluster_map.get(node, 0) for node in G.nodes()]
                node_trace = go.Scatter(
                    x=[pos[node][0] for node in G.nodes()],
                    y=[pos[node][1] for node in G.nodes()],
                    text=list(G.nodes()),
                    mode='markers+text',
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=node_colors,
                        colorscale='Set3',
                        colorbar=dict(title="Clusters"),
                        line=dict(width=2, color='darkblue')
                    ),
                    hoverinfo='text'
                )

                fig_net = go.Figure(
                    data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title="User Network (Click & Drag to Explore)",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=60),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    )
                )
                st.plotly_chart(fig_net, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Network graph failed: {e}")
        st.info("Try reducing the number of influencers in the network or check data quality.")

    # ----------------------------
    # ⚠️ High-Risk Influencers
    # ----------------------------
    st.markdown("### ⚠️ High-Risk Influencers")
    st.markdown("**High-Risk Influencers**: Highlights accounts involved in 3 or more coordinated messages, indicating potential amplification roles.")
    try:
        if 'sim_df' in locals() and not sim_df.empty:
            all_influencers = pd.concat([
                sim_df[['account_id1']].rename(columns={'account_id1': 'account_id'}),
                sim_df[['account_id2']].rename(columns={'account_id2': 'account_id'})
            ])['account_id'].dropna().astype(str)
            influencer_counts = all_influencers.value_counts()
            high_risk = influencer_counts[influencer_counts >= 3]
            if not high_risk.empty:
                fig_hr = px.bar(
                    high_risk,
                    title="Influencers in ≥3 Coordinated Messages",
                    labels={'value': 'Coordination Instances', 'index': 'Influencer'},
                    color='value',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_hr, use_container_width=True)
            else:
                st.info("No influencers found participating in 3 or more coordinated messages.")
        else:
            st.info("No coordinated narratives detected to identify high-risk influencers.")
    except Exception as e:
        st.warning(f"Risk analysis failed: {e}")
