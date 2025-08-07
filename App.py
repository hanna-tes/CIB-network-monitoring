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
from io import StringIO

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
    """Cleans text by removing RT/QT prefixes, @mentions, URLs, and normalizing spaces. Used for similarity analysis."""
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

    # Rename columns for plotting later
    df_processed.rename(columns={'account_id': 'Influencer'}, inplace=True)
    df_processed.rename(columns={'object_id': 'text'}, inplace=True)
    df_processed.rename(columns={'timestamp_share': 'Timestamp'}, inplace=True)

    # Ensure optional columns exist
    if 'Outlet' not in df_processed.columns:
        df_processed['Outlet'] = np.nan
    if 'Channel' not in df_processed.columns:
        df_processed['Channel'] = np.nan

    # Convert Timestamp to datetime (UTC)
    df_processed['Timestamp'] = pd.to_datetime(df_processed['Timestamp'], unit='s', utc=True)

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
    influencer_column = 'Influencer'
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
    clean_df = df[['original_text', 'Influencer', 'Timestamp', 'Platform', 'URL']].copy()
    clean_df['original_text'] = clean_df['original_text'].astype(str)
    clean_df = clean_df.dropna(subset=['original_text', 'Influencer', 'Timestamp'])
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
            'shared_narrative_snippet': snippet,
            'similarity_score': round(sim_matrix[i, j], 3),
            'account_id_1': row1['Influencer'],
            'platform_1': row1['Platform'],
            'timestamp_1': row1['Timestamp'],
            'url_1': row1['URL'],
            'account_id_2': row2['Influencer'],
            'platform_2': row2['Platform'],
            'timestamp_2': row2['Timestamp'],
            'url_2': row2['URL'],
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
data_source_option = st.sidebar.radio(
    "Choose data source:",
    ("Use Default Datasets", "Upload Social Media CSVs")  # REMOVED: "Upload Radar Leads CSV"
)

# Coordination Mode Selector
st.sidebar.header("🎯 Coordination Analysis Mode")
coordination_mode = st.sidebar.radio(
    "Analyze coordination by:",
    ("Text Content", "Shared URLs"),
    help="Choose what defines a coordinated action: similar text messages or sharing the same external link."
)

# Clear cache when mode or source changes
if 'last_data_source_option' not in st.session_state or st.session_state.last_data_source_option != data_source_option:
    st.cache_data.clear()
    st.session_state.last_data_source_option = data_source_option
if 'last_coordination_mode' not in st.session_state or st.session_state.last_coordination_mode != coordination_mode:
    st.cache_data.clear()
    st.session_state.last_coordination_mode = coordination_mode

# Initialize data variables
combined_raw_df = pd.DataFrame()
data_source_type = "default"  # ✅ Critical: Ensure this is defined before use

# Function to read uploaded CSV with various encodings
def read_uploaded_file(uploaded_file, file_name):
    if not uploaded_file:
        return pd.DataFrame()
    bytes_data = uploaded_file.getvalue()
    encodings = ['utf-8', 'utf-8-sig', 'utf-16le', 'utf-16be', 'utf-16', 'latin1', 'cp1252']
    decoded_content = None
    detected_enc = None
    for enc in encodings:
        try:
            decoded_content = bytes_data.decode(enc)
            detected_enc = enc
            st.sidebar.info(f"✅ {file_name}: Decoded using '{enc}'")
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    
    if decoded_content is None:
        st.error(f"❌ Failed to read {file_name} CSV: Could not decode with any supported encoding.")
        return pd.DataFrame()
    
    # Use the first few lines to detect a tab-separated file
    sample_lines = decoded_content.strip().splitlines()[:5]
    is_tsv = any('\t' in line for line in sample_lines)
    sep = '\t' if is_tsv else ','
    
    try:
        df = pd.read_csv(StringIO(decoded_content), sep=sep, low_memory=False)
        st.sidebar.success(f"✅ {file_name}: Loaded {len(df)} rows (sep='{sep}', enc='{detected_enc}')")
        return df
    except Exception as e:
        st.error(f"❌ Failed to parse {file_name} CSV after decoding: {e}")
        return pd.DataFrame()

# Load data based on sidebar selection
if data_source_option == "Use Default Datasets":
    st.sidebar.info("Using default datasets from GitHub.")
    with st.spinner("📥 Loading and combining default datasets..."):
        base_url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/"
        urls = {
            "meltwater": f"{base_url}TogoJULYData%20-%20Sheet1.csv",
            "civicsignals": f"{base_url}togo-or-lome-or-togo-all-story-urls-20250707142808.csv"
        }
        meltwater_df = pd.DataFrame()
        civicsignals_df = pd.DataFrame()
        results = []
        for key, url in urls.items():
            try:
                df = pd.read_csv(url, sep=',')
                if not df.empty:
                    if key == "meltwater":
                        meltwater_df = df
                    elif key == "civicsignals":
                        civicsignals_df = df
                    results.append({"Source": key.capitalize(), "Rows Loaded": len(df)})
                else:
                    results.append({"Source": key.capitalize(), "Rows Loaded": 0})
            except Exception as e:
                results.append({"Source": key.capitalize(), "Rows Loaded": "Error"})
        
        # Show data overview
        if results:
            results_df = pd.DataFrame(results)
            st.sidebar.markdown("### 📊 Data Overview")
            st.sidebar.dataframe(results_df, hide_index=True)

        obj_map = {
            "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
            "civicsignals": "title" if coordination_mode == "Text Content" else "url"
        }
        combined_raw_df = combine_social_media_data(
            meltwater_df if not meltwater_df.empty else None,
            civicsignals_df if not civicsignals_df.empty else None,
            None,
            meltwater_object_col=obj_map["meltwater"],
            civicsignals_object_col=obj_map["civicsignals"]
        )
        data_source_type = "default"  # ✅ Overwrite

elif data_source_option == "Upload Social Media CSVs":
    st.sidebar.info("Upload your CSV files below.")
    uploaded_meltwater = st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater_upload")
    uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
    uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")
    
    meltwater_df_upload = read_uploaded_file(uploaded_meltwater, "Meltwater")
    civicsignals_df_upload = read_uploaded_file(uploaded_civicsignals, "CivicSignals")
    openmeasure_df_upload = read_uploaded_file(uploaded_openmeasure, "Open-Measure")
    
    with st.spinner("📥 Combining uploaded datasets..."):
        obj_map = {
            "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
            "civicsignals": "title" if coordination_mode == "Text Content" else "url",
            "openmeasure": "text" if coordination_mode == "Text Content" else "url"
        }
        combined_raw_df = combine_social_media_data(
            meltwater_df_upload,
            civicsignals_df_upload,
            openmeasure_df_upload,
            meltwater_object_col=obj_map["meltwater"],
            civicsignals_object_col=obj_map["civicsignals"],
            openmeasure_object_col=obj_map["openmeasure"]
        )
        data_source_type = "uploaded_social"  # ✅ Overwrite

# Exit if no data
if combined_raw_df is None or combined_raw_df.empty:
    st.warning("No data available. Please upload a CSV file or check the default datasets.")
    st.stop()

# === Show Top 10 Data Preview ===
st.markdown("### 🔍 Processed Data Preview (Top 10 Rows)")
preview_cols = ['Influencer', 'text', 'Timestamp', 'URL', 'Platform']
available_cols = [col for col in preview_cols if col in df.columns]
st.dataframe(df[available_cols].head(10), height=300)

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

download_df_columns = ['Influencer', 'text', 'Timestamp', 'URL']
downloadable_df = df[download_df_columns].copy() if all(col in df.columns for col in download_df_columns) else pd.DataFrame()

if not downloadable_df.empty:
    combined_preprocessed_csv = convert_df_to_csv(downloadable_df)
    st.sidebar.download_button(
        "Download Preprocessed Dataset (Core Columns)",
        combined_preprocessed_csv,
        f"preprocessed_combined_core_data_{coordination_mode.replace(' ', '_').lower()}.csv",
        "text/csv",
        help="Downloads the data after all preprocessing and column mapping. 'text' contains either text or URL based on your selection."
    )
else:
    st.sidebar.warning("Could not create downloadable dataset with core columns.")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters (Apply to all tabs)")
if 'Timestamp' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
    st.error("Timestamp must be a datetime column.")
    st.stop()

min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()
selected_date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

if len(selected_date_range) == 2:
    start_dt = pd.Timestamp(selected_date_range[0], tz='UTC')
    end_dt = pd.Timestamp(selected_date_range[1], tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
elif len(selected_date_range) == 1:
    start_dt = pd.Timestamp(selected_date_range[0], tz='UTC')
    end_dt = start_dt + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
else:
    start_dt = df['Timestamp'].min()
    end_dt = df['Timestamp'].max()

available_platforms = df['Platform'].dropna().astype(str).unique().tolist()
platforms = st.sidebar.multiselect("Platforms", options=available_platforms, default=available_platforms)

filtered_df_global = df[
    (df['Timestamp'] >= start_dt) &
    (df['Timestamp'] <= end_dt) &
    (df['Platform'].isin(platforms))
].copy()

if filtered_df_global.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# Export filtered data
st.sidebar.markdown("### 📄 Export Filtered Results")
filtered_csv_data = convert_df_to_csv(filtered_df_global)
st.sidebar.download_button("Download Filtered Data (All Columns)", filtered_csv_data, "filtered_dashboard_data.csv", "text/csv")

# --- TABS ---
tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "🔍 Analysis",
    "🌐 Network & Risk",
])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Summary Statistics")
    
    # === Show Core Semantic Data Preview ===
    # === Show Core Semantic Data Preview ===
st.markdown("### 🔬 Core Data Preview (Raw Semantic Columns)")
st.markdown(f"**Data Source:** `{data_source_option}` | **Coordination Mode:** `{coordination_mode}` | **Total Rows:** `{len(combined_raw_df):,}`")

# Define core columns
core_columns = ['account_id', 'content_id', 'object_id', 'timestamp_share']
available_core_cols = [col for col in core_columns if col in combined_raw_df.columns]

if not combined_raw_df.empty and available_core_cols:
    # Show top 10 rows of core columns
    st.dataframe(combined_raw_df[available_core_cols].head(10), height=300)
else:
    st.info("No core data available to display (account_id, content_id, object_id, timestamp_share).")

    # Optional: Show source dataset distribution
    if 'source_dataset' in combined_raw_df.columns:
        st.markdown("### 📊 Data Sources")
        source_counts = combined_raw_df['source_dataset'].value_counts()
        st.dataframe(source_counts)

    # === Aggregated Visuals (after preprocessing) ===
    if not filtered_df_global.empty:
        if 'Influencer' in filtered_df_global.columns:
            top_influencers = filtered_df_global['Influencer'].value_counts().head(10)
            fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'Influencer'})
            st.plotly_chart(fig_src, use_container_width=True)
        else:
            st.info("No 'Influencer' column found.")

        if 'Platform' in filtered_df_global.columns and not filtered_df_global['Platform'].empty:
            top_platforms = filtered_df_global['Platform'].value_counts().head(10)
            fig_platform = px.bar(top_platforms, title="Top 10 Platforms", labels={'value': 'Posts', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True)
        else:
            st.info("No 'Platform' column found or no data for platforms.")

        if 'text' in filtered_df_global.columns and not filtered_df_global['text'].empty:
            filtered_df_temp = filtered_df_global.copy()
            filtered_df_temp['hashtags'] = filtered_df_temp['text'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
            all_hashtags = [tag for tags_list in filtered_df_temp['hashtags'] if isinstance(tags_list, list) for tag in tags_list if tags_list]
            if all_hashtags:
                hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags", labels={'value': 'Frequency', 'index': 'Hashtag'})
                st.plotly_chart(fig_ht, use_container_width=True)
            else:
                st.info("No hashtags found in the filtered data 'text' column.")
        else:
            st.info("No 'text' column found or it's empty to extract hashtags.")

        time_series = filtered_df_global.set_index('Timestamp').resample('D').size()
        fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'Timestamp': 'Date'})
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No data available to display summary statistics.")
        
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
        # Now 'object_id' still exists
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
                sim_df = cached_similarity_analysis(analysis_df, threshold=0.85, data_source=data_source_type)

            if not sim_df.empty:
                st.success(f"✅ Found {len(sim_df)} similar pairs.")

                # Label similarity levels
                sim_df['similarity_level'] = sim_df['similarity_score'].apply(
                    lambda x: "🚨 Exact Copy / Bot-Level" if x >= 0.98 else
                              "🔥 Near-Identical Coordination" if x >= 0.95 else
                              "🟡 Highly Similar Messaging" if x >= 0.90 else
                              "🟢 Loosely Similar"
                )

                # Sort by similarity
                sim_df_sorted = sim_df.sort_values('similarity_score', ascending=False).reset_index(drop=True)

                # Add visual display
                sim_df_sorted['similarity_display'] = sim_df_sorted['similarity_score'].apply(
                    lambda x: f"**{x}**" + (" 🔴" if x >= 0.98 else " 🟠" if x >= 0.95 else " 🟡" if x >= 0.90 else " 🟢")
                )

                # Show high-similarity pairs
                st.markdown("### 🔝 Top High-Similarity Pairs (≥ 0.95)")
                high_sim = sim_df_sorted[sim_df_sorted['similarity_score'] >= 0.95]
                if not high_sim.empty:
                    st.dataframe(
                        high_sim[[
                            'similarity_display', 'similarity_level', 'shared_narrative_snippet',
                            'account_id_1', 'platform_1', 'account_id_2', 'platform_2'
                        ]],
                        use_container_width=True
                    )
                else:
                    st.info("No near-identical content (≥0.95) found.")

                st.markdown("### 🔎 All Similar Pairs")
                st.dataframe(
                    sim_df_sorted[[
                        'similarity_display', 'similarity_level', 'shared_narrative_snippet',
                        'account_id_1', 'platform_1', 'account_id_2', 'platform_2', 'platforms_involved'
                    ]],
                    use_container_width=True
                )
            else:
                st.info("No significant similarities found above threshold.")

    elif coordination_mode == "Shared URLs":
        st.markdown("### 🔗 URL-Based Coordination Analysis")
        url_counts = filtered_df_global['URL'].value_counts()
        url_coord_df = url_counts[url_counts > 1].to_frame('share_count').reset_index()
        url_coord_df.rename(columns={'index': 'URL'}, inplace=True)
        
        if not url_coord_df.empty:
            st.success(f"✅ Found {len(url_coord_df)} URLs shared by multiple accounts.")
            
            def get_sharers(url):
                sharers = filtered_df_global[filtered_df_global['URL'] == url]['account_id'].unique()
                platforms = filtered_df_global[filtered_df_global['URL'] == url]['Platform'].unique()
                return ", ".join(sharers), ", ".join(platforms)
            
            url_coord_df[['accounts_sharing', 'platforms_involved']] = url_coord_df['URL'].apply(
                lambda x: pd.Series(get_sharers(x))
            )

            url_coord_df['platforms_involved_count'] = url_coord_df['platforms_involved'].apply(
                lambda x: len(set(x.split(', '))) if x else 0
            )

            st.dataframe(url_coord_df.sort_values('share_count', ascending=False))
            
            @st.cache_data
            def convert_df_to_csv_url(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv_url(url_coord_df)
            st.download_button(
                "Download URL Coordination Data",
                csv,
                "url_coordination_results.csv",
                "text/csv",
                key='download-url-csv'
            )
        else:
            st.info("No URLs were shared by more than one account in the filtered dataset.")
            
# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.header("🌐 Network & Risk")
    st.markdown("""
        This section provides an in-depth look at the network of user interactions and identifies potential risks. 
        You can analyze coordination based on shared text content or shared URLs to visualize how users are connected.
    """)
    st.subheader("🚨 High-Risk Accounts & Networks")

    # --- Clustering or URL Analysis ---
    if coordination_mode == "Text Content":
        df_for_clustering = filtered_df_global[filtered_df_global['text'].astype(str).str.strip() != ""].copy()
        if df_for_clustering.empty:
            st.info("No valid text data for clustering analysis.")
            clustered_df = pd.DataFrame()
        else:
            clustered_df = cached_clustering(df_for_clustering, data_source=data_source_type)
        
        if 'cluster' not in clustered_df.columns:
            st.warning("⚠️ Clustering did not return 'cluster' column.")
            clustered_df['cluster'] = -1

        if not clustered_df.empty:
            cluster_counts = clustered_df['cluster'].value_counts()
            if not cluster_counts.empty:
                st.markdown("### 🤖 Detected Coordination Clusters")
                fig_clust = px.bar(
                    cluster_counts,
                    title="Cluster Sizes",
                    labels={'value': 'Member Count', 'index': 'Cluster ID'},
                    color=cluster_counts.index.astype(str),
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_clust, use_container_width=True)
                st.dataframe(clustered_df[['Influencer', 'text', 'Timestamp', 'cluster']])
            else:
                st.info("No clusters detected.")
        else:
            st.info("No data available for clustering.")

    elif coordination_mode == "Shared URLs":
        url_groups = filtered_df_global.groupby('URL').filter(lambda x: len(x) > 1)
        if not url_groups.empty:
            st.success(f"✅ Found {url_groups['URL'].nunique()} URLs shared by multiple accounts.")
            st.dataframe(url_groups.groupby('URL').agg(
                post_count=('Influencer', 'size'),
                accounts_involved=('Influencer', lambda x: ', '.join(x.unique())),
                platforms_involved=('Platform', lambda x: ', '.join(x.unique())),
                first_share=('Timestamp', 'min'),
                last_share=('Timestamp', 'max')
            ).sort_values('post_count', ascending=False).reset_index())
        else:
            st.info("No URLs were shared by more than one account.")

    # --- Network Graph ---
    st.markdown("### 🕸️ Coordinated Network Graph")
    MAX_NETWORK_NODES = st.slider(
        "Max nodes to display in network graph (for performance)",
        10, 500, 100,
        key="max_network_nodes_tab3"
    )

    G = nx.Graph()
    filtered_df_for_graph = filtered_df_global.copy()

    # Build graph based on coordination mode
    if coordination_mode == "Text Content":
        if 'clustered_df' in locals() and not clustered_df.empty:
            top_accounts = clustered_df['Influencer'].value_counts().head(MAX_NETWORK_NODES).index
            filtered_df_for_graph = clustered_df[clustered_df['Influencer'].isin(top_accounts)].copy()
            G, pos, cluster_map = cached_network_graph(filtered_df_for_graph, "text", data_source=data_source_type)
    elif coordination_mode == "Shared URLs":
        top_accounts = filtered_df_global['Influencer'].value_counts().head(MAX_NETWORK_NODES).index
        filtered_df_for_graph = filtered_df_global[filtered_df_global['Influencer'].isin(top_accounts)].copy()
        G, pos, cluster_map = cached_network_graph(filtered_df_for_graph, "url", data_source=data_source_type)

    st.info(f"👥 Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if G.number_of_nodes() == 0:
        st.info("Not enough data to build a network graph.")
    else:
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() == 0:
            st.info("No connected nodes after filtering isolates.")
        else:
            # Recompute layout with better spacing
            pos = nx.spring_layout(G, seed=42, k=0.7, iterations=100)

            # Node sizes based on degree
            node_weights = dict(G.degree(weight='weight'))
            min_size, max_size = 15, 50
            node_sizes = [
                min_size + (node_weights[n] - min(node_weights.values())) /
                (max(node_weights.values()) - min(node_weights.values()) + 1e-6) * (max_size - min_size)
                for n in G.nodes()
            ]

            # Colors by cluster
            colors = [cluster_map.get(n, -2) for n in G.nodes()]

            # Edge traces
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.8, color='#aaa'),
                hoverinfo='none',
                showlegend=False
            )

            # Node positions
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]

            # Hover text
            node_text = [
                f"<b>{node}</b><br>"
                f"Connections: {G.degree(node)}<br>"
                f"Platform: {G.nodes[node].get('platform', 'Unknown')}<br>"
                f"Cluster: {G.nodes[node].get('cluster', 'N/A')}"
                for node in G.nodes()
            ]

            # Truncate labels for readability
            node_labels = [n if len(n) <= 12 else n[:10] + "..." for n in G.nodes()]

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=node_labels,
                textposition="top center",
                textfont=dict(size=10, color='black'),
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    size=node_sizes,
                    color=colors,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Cluster ID", thickness=10, x=1.0, len=0.5),
                    line=dict(width=1.5, color='white')
                ),
                showlegend=False
            )

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"<b>{coordination_mode} Coordination Network</b><br><sup>{G.number_of_nodes()} nodes, {G.number_of_edges()} edges</sup>",
                    titlefont=dict(size=14),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(l=20, r=20, b=40, t=60),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white',
                    height=600,
                    width=None
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- High-Risk Influencers ---
    st.markdown("### ⚠️ High-Risk Influencers")
    try:
        if 'sim_df' in locals() and not sim_df.empty:
            all_influencers = pd.concat([
                sim_df[['account_id1']].rename(columns={'account_id1': 'Influencer'}),
                sim_df[['account_id2']].rename(columns={'account_id2': 'Influencer'})
            ])['Influencer'].dropna().astype(str)
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
