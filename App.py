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
    # Map to old column names for existing visuals
    df_processed.rename(columns={'account_id': 'Influencer'}, inplace=True)
    df_processed.rename(columns={'object_id': 'text'}, inplace=True)
    df_processed.rename(columns={'timestamp_share': 'Timestamp'}, inplace=True)
    
    if 'Outlet' not in df_processed.columns:
        df_processed['Outlet'] = np.nan
    if 'Channel' not in df_processed.columns:
        df_processed['Channel'] = np.nan
    
    # Re-create 'Timestamp' as datetime object
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
            'text1': row1['original_text'],
            'account_id1': row1['Influencer'],
            'platform1': row1['Platform'],
            'timestamp_share1': row1['Timestamp'],
            'url1': row1['URL'],
            'text2': row2['original_text'],
            'account_id2': row2['Influencer'],
            'platform2': row2['Platform'],
            'timestamp_share2': row2['Timestamp'],
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
data_source_option = st.sidebar.radio(
    "Choose data source:",
    ("Use Default Datasets", "Upload Social Media CSVs", "Upload Radar Leads CSV")
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

combined_raw_df = pd.DataFrame()
data_source_type = "default"

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
            "civicsignals": "title" if coordination_mode == "Text Content" else "url"
        }
        combined_raw_df = combine_social_media_data(
            meltwater_df if not meltwater_df.empty else None,
            civicsignals_df if not civicsignals_df.empty else None,
            None,  # No Open-Measure by default
            meltwater_object_col=obj_map["meltwater"],
            civicsignals_object_col=obj_map["civicsignals"]
        )
        data_source_type = "default"

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
        data_source_type = "uploaded_social"

elif data_source_option == "Upload Radar Leads CSV":
    st.sidebar.info("Upload a single CSV file with 'Country', 'Leads', and 'URL' columns.")
    uploaded_radar_file = st.sidebar.file_uploader("Upload Radar Leads CSV", type=["csv"], key="radar_upload")
    if uploaded_radar_file:
        with st.spinner("📥 Loading and processing Radar Leads file..."):
            radar_df_raw = read_uploaded_file(uploaded_radar_file, "Radar Leads")
            if not radar_df_raw.empty:
                radar_df_raw.columns = radar_df_raw.columns.str.lower()
                required_cols = ['country', 'leads', 'url']
                if all(col in radar_df_raw.columns for col in required_cols):
                    combined_raw_df = pd.DataFrame({
                        'account_id': radar_df_raw['country'].astype(str),
                        'content_id': radar_df_raw['leads'].astype(str),
                        'object_id': radar_df_raw['leads'].astype(str),
                        'original_url': radar_df_raw['url'].astype(str),
                        'timestamp_share': pd.to_datetime('now', utc=True).timestamp(),
                        'source_dataset': 'Radar Leads'
                    })
                    st.sidebar.success(f"✅ Radar Leads file loaded with {len(combined_raw_df)} rows.")
                    st.sidebar.warning("⚠️ No Timestamp column found. Using current time as a placeholder.")
                else:
                    st.error(f"❌ Radar Leads file is missing required columns. Ensure it has 'Country', 'Leads', and 'URL'.")
                    combined_raw_df = pd.DataFrame()
        data_source_type = "uploaded_radar"

# Exit if no data
if combined_raw_df is None or combined_raw_df.empty:
    st.warning("No data available. Please upload a CSV file or check the default URL.")
    st.stop()

# Debug
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Mode:** `{coordination_mode}`")
st.sidebar.markdown(f"**Source:** `{data_source_option}`")
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Analysis",
    "🌐 Network & Risk",
    "📰 Narrative Dashboard",
])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Summary Statistics")
    if filtered_df_global.empty:
        st.info("No data available to display summary statistics.")
    else:
        # Top 10 Influencers
        if 'Influencer' in filtered_df_global.columns:
            top_influencers = filtered_df_global['Influencer'].value_counts().head(10)
            if len(top_influencers) > 0:
                fig_src = px.bar(
                    top_influencers,
                    title="Top 10 Influencers",
                    labels={'value': 'Posts', 'index': 'Influencer'}
                )
                st.plotly_chart(fig_src, use_container_width=True)
            else:
                st.info("No influencer data to display.")
        else:
            st.info("No 'Influencer' column found.")

        # Top 10 Platforms
        if 'Platform' in filtered_df_global.columns:
            platform_counts = filtered_df_global['Platform'].value_counts().head(10)
            if len(platform_counts) > 0:
                fig_platform = px.bar(
                    platform_counts,
                    title="Top 10 Platforms",
                    labels={'value': 'Posts', 'index': 'Platform'}
                )
                st.plotly_chart(fig_platform, use_container_width=True)
            else:
                st.info("No platform data to display.")
        else:
            st.info("No 'Platform' column found.")

        # Top 10 Channels
        if 'Channel' in filtered_df_global.columns:
            channel_counts = filtered_df_global['Channel'].value_counts().dropna().head(10)
            if len(channel_counts) > 0:
                fig_chan = px.bar(
                    channel_counts,
                    title="Top 10 Channels",
                    labels={'value': 'Posts', 'index': 'Channel'}
                )
                st.plotly_chart(fig_chan, use_container_width=True)
            else:
                st.info("No channel data to display.")
        # No else: silently skip if column doesn't exist (per your request)

        # Top 10 Hashtags
        if 'text' in filtered_df_global.columns:
            text_data = filtered_df_global['text'].astype(str)
            hashtags_series = text_data.str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
            all_hashtags = [tag for tags_list in hashtags_series if isinstance(tags_list, list) for tag in tags_list]
            if all_hashtags:
                hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                fig_ht = px.bar(
                    hashtag_counts,
                    title="Top 10 Hashtags",
                    labels={'value': 'Frequency', 'index': 'Hashtag'}
                )
                st.plotly_chart(fig_ht, use_container_width=True)
            else:
                st.info("No hashtags found in the 'text' column.")
        # No else: silently skip if no 'text' column

        # Daily Post Volume
        if 'Timestamp' in filtered_df_global.columns and pd.api.types.is_datetime64_any_dtype(filtered_df_global['Timestamp']):
            time_series = filtered_df_global.set_index('Timestamp').resample('D').size()
            if len(time_series) > 0:
                fig_ts = px.area(
                    time_series,
                    title="Daily Post Volume",
                    labels={'value': 'Number of Posts', 'Timestamp': 'Date'}
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("No timestamp data to display daily volume.")
        else:
            st.info("No valid timestamp data for time series.")
            
# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.header("🔍 Analysis")
    st.markdown("""
        This section helps identify coordinated narratives by analyzing the textual similarity between posts. 
        It looks for pairs of posts that are highly similar to each other, which can be an indicator of 
        coordinated information operations.
    """)
    st.subheader("🧠 Narrative Detection & Coordination")
    MAX_ROWS = st.sidebar.slider("Max posts to analyze for similarity", 100, 1000, 300)
    analysis_df = filtered_df_global[filtered_df_global['original_text'].astype(str).str.strip() != ""].head(MAX_ROWS).copy()
    if analysis_df.empty:
        st.info("No valid text data available for similarity analysis after applying filters and row limit.")
    else:
        with st.spinner(f"🔍 Finding coordinated narratives among {len(analysis_df)} posts..."):
            sim_df = cached_similarity_analysis(analysis_df, threshold=0.85, data_source=data_source_type)
        if not sim_df.empty:
            st.success(f"✅ Found {len(sim_df)} similar pairs.")
            narrative_summary = sim_df.groupby('shared_narrative').agg(
                share_count=('similarity', 'count'),
                influencers_involved=('account_id1', lambda x: ", ".join(x.astype(str).unique()[:5]) + ("..." if len(x.unique()) > 5 else ""))
            ).sort_values(by='share_count', ascending=False).reset_index()
            st.markdown("### 🔝 Top Coordinated Narratives")
            fig_nar = px.bar(
                narrative_summary.head(10), x='share_count', y='shared_narrative', orientation='h',
                title="Top 10 Most Shared Narratives", labels={'shared_narrative': 'Narrative Snippet', 'share_count': 'Share Count'},
                color='share_count', color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_nar, use_container_width=True)
            st.dataframe(narrative_summary)
            st.markdown("### 🔄 Full Similarity Pairs")
            st.dataframe(sim_df.drop(columns=['shared_narrative'], errors='ignore'))
        else:
            st.info("No significant similarities found above threshold.")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.header("🌐 Network & Risk")
    st.markdown("""
        This section identifies high-risk coordination networks by analyzing shared content or URLs.
        Only significant clusters and influential nodes are displayed to ensure clarity and relevance.
    """)

    st.subheader("🚨 High-Risk Accounts & Networks")

    # === Clustering Analysis (Text) or URL Sharing ===
    if coordination_mode == "Text Content":
        if 'original_text' not in filtered_df_global.columns:
            st.info("No text data available for clustering.")
        else:
            df_for_clustering = filtered_df_global[filtered_df_global['original_text'].str.strip() != ""].copy()
            if df_for_clustering.empty:
                st.info("No valid text data for clustering after filtering.")
            else:
                clustered_df = cached_clustering(df_for_clustering, data_source=data_source_type)
                if 'cluster' not in clustered_df.columns:
                    st.info("Clustering failed or no clusters found.")
                else:
                    # Focus only on meaningful clusters (size >= 2, not noise)
                    cluster_counts = clustered_df['cluster'].value_counts()
                    meaningful_clusters = cluster_counts[(cluster_counts.index != -1) & (cluster_counts >= 2)]
                    
                    if meaningful_clusters.empty:
                        st.info("No significant coordination clusters detected (min. 2 members).")
                    else:
                        st.markdown("### 🤖 Significant Coordination Clusters")
                        fig_clust = px.bar(
                            meaningful_clusters,
                            title=f"{len(meaningful_clusters)} Clusters with ≥2 Members",
                            labels={'value': 'Member Count', 'index': 'Cluster ID'},
                            color=meaningful_clusters.index.astype(str),
                            color_discrete_sequence=px.colors.qualitative.Bold,
                        )
                        fig_clust.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=40),
                            showlegend=False,
                            xaxis={'tickmode': 'linear'} if len(meaningful_clusters) < 10 else {}
                        )
                        st.plotly_chart(fig_clust, use_container_width=True)

                        # Show only top 5 clusters
                        top_clusters = meaningful_clusters.head(5).index
                        st.dataframe(
                            clustered_df[clustered_df['cluster'].isin(top_clusters)][
                                ['Influencer', 'text', 'Timestamp', 'cluster']
                            ].reset_index(drop=True),
                            height=250
                        )
    elif coordination_mode == "Shared URLs":
        if 'URL' not in filtered_df_global.columns:
            st.info("No URL data available.")
        else:
            url_groups = filtered_df_global[filtered_df_global['URL'].str.strip() != ""]
            shared_urls = url_groups.groupby('URL').filter(lambda x: len(x) > 1)
            if shared_urls.empty:
                st.info("No URLs shared by multiple accounts.")
            else:
                st.success(f"🔗 Found **{shared_urls['URL'].nunique()}** URLs shared by ≥2 accounts.")
                url_summary = shared_urls.groupby('URL').agg(
                    share_count=('Influencer', 'size'),
                    unique_accounts=('Influencer', 'nunique'),
                    platforms=('Platform', lambda x: ', '.join(x.unique())),
                    first_seen=('Timestamp', 'min'),
                    last_seen=('Timestamp', 'max')
                ).sort_values('share_count', ascending=False).head(10)
                st.dataframe(url_summary)

    # ==================== IMPROVED NETWORK GRAPH ====================
    st.markdown("### 🕸️ Key Coordination Network")

    # Limit nodes for clarity and performance
    MAX_NODES = st.slider(
        "Max influential nodes to display",
        min_value=10, max_value=200, value=50,
        help="Limits graph to top accounts by interaction frequency for clarity."
    )

    # Build graph only if we have data
    G = nx.Graph()
    filtered_df_for_graph = filtered_df_global.copy()

    if coordination_mode == "Text Content" and 'clustered_df' in locals():
        # Use only top influencers from significant clusters
        top_accounts = clustered_df['Influencer'].value_counts().head(MAX_NODES).index
        filtered_df_for_graph = clustered_df[clustered_df['Influencer'].isin(top_accounts)]
        G, pos, cluster_map = cached_network_graph(filtered_df_for_graph, "text", data_source=data_source_type)
    elif coordination_mode == "Shared URLs":
        top_accounts = filtered_df_global['Influencer'].value_counts().head(MAX_NODES).index
        filtered_df_for_graph = filtered_df_global[filtered_df_global['Influencer'].isin(top_accounts)]
        G, pos, cluster_map = cached_network_graph(filtered_df_for_graph, "url", data_source=data_source_type)
    else:
        st.info("No data available for network graph.")
        G = nx.Graph()

    if G.number_of_nodes() == 0:
        st.info("No network connections to display.")
    else:
        # Filter: Only keep nodes with degree >= 1 (remove isolated nodes)
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() == 0:
            st.info("No connected nodes after filtering isolates.")
        else:
            # Recompute layout
            pos = nx.spring_layout(G, seed=42, k=0.5, iterations=100)

            # Node size by weighted degree
            node_weights = dict(G.degree(weight='weight'))
            min_size, max_size = 12, 40
            sizes = [min_size + (node_weights[n] - min(node_weights.values())) /
                     (max(node_weights.values()) - min(node_weights.values()) + 1e-6) * (max_size - min_size)
                     for n in G.nodes()]

            # Colors by cluster
            colors = [cluster_map.get(n, -2) for n in G.nodes()]
            unique_clusters = len(set(colors))

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

            # Node trace
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]

            node_text = [
                f"<b>{node}</b><br>Connections: {G.degree(node)}<br>Platform: {G.nodes[node].get('platform', 'Unknown')}"
                for node in G.nodes()
            ]

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=[node[:10] + "..." if len(node) > 10 else node for node in G.nodes()],
                textposition="top center",
                textfont=dict(size=9),
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale='Viridis',
                    showscale=unique_clusters > 1,
                    colorbar=dict(
                        title="Cluster",
                        thickness=10,
                        x=1.0,
                        len=0.5
                    ) if unique_clusters > 1 else None,
                    line=dict(width=1.5, color='white')
                ),
                showlegend=False
            )

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"<b>{coordination_mode} Network</b><br><sup>{G.number_of_nodes()} nodes, {G.number_of_edges()} edges</sup>",
                    titlefont=dict(size=14),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(l=20, r=20, b=40, t=60),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white',
                    height=500,
                    width=None  # Responsive width
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    # ==================== High-Risk Influencers ====================
    st.markdown("### ⚠️ High-Risk Influencers")
    try:
        if 'sim_df' in locals() and not sim_df.empty:
            all_influencers = pd.concat([
                sim_df['account_id1'],
                sim_df['account_id2']
            ]).value_counts()
            high_risk = all_influencers[all_influencers >= 3]
            if not high_risk.empty:
                fig_hr = px.bar(
                    high_risk.head(10),
                    title="Top Influencers in ≥3 Coordinated Pairs",
                    labels={'value': 'Coordination Count', 'index': 'Influencer'},
                    color='value',
                    color_continuous_scale='Reds'
                )
                fig_hr.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=40))
                st.plotly_chart(fig_hr, use_container_width=True)
            else:
                st.info("No influencer involved in 3+ coordinated messages.")
        else:
            st.info("No coordinated narratives detected.")
    except Exception as e:
        st.warning(f"Could not compute risk scores: {e}")
# ==================== TAB 4: Narrative Dashboard ====================
with tab4:
    st.header("📰 Pre-Clustered Narrative Dashboard")
    st.markdown("""
        This dashboard is designed for journalists to quickly gain insights from a pre-processed and 
        summarized dataset of narratives. Upload a CSV file that already contains clustered posts 
        to visualize the key narratives, their virality, and their geographical distribution.
    """)
    st.subheader("📰 Pre-Clustered Narrative Analysis")
    narrative_file = st.file_uploader("Upload pre-processed Narrative CSV", type=["csv"], key="narrative_uploader")
    if narrative_file:
        try:
            narrative_df = pd.read_csv(narrative_file)
            st.success("✅ Narrative data loaded successfully!")
            required_cols = ['Country', 'Leads', 'Emerging Virality', 'Evidence']
            if not all(col in narrative_df.columns for col in required_cols):
                st.error(f"❌ Uploaded file is missing one or more required columns: {required_cols}. Please check your data.")
            else:
                st.markdown("### 📊 Narrative Virality Distribution")
                virality_counts = narrative_df['Emerging Virality'].value_counts().sort_index()
                fig_virality = px.bar(
                    virality_counts,
                    title="Distribution of Narrative Virality",
                    labels={'value': 'Number of Narratives', 'index': 'Virality Level'},
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig_virality, use_container_width=True)
                st.markdown("### 🌍 Narratives by Country")
                country_counts = narrative_df['Country'].value_counts()
                fig_country = px.pie(
                    country_counts,
                    values=country_counts.values,
                    names=country_counts.index,
                    title="Narratives by Country",
                    hole=.3
                )
                st.plotly_chart(fig_country, use_container_width=True)
                st.markdown("### 📋 Narrative Clusters Table")
                st.dataframe(narrative_df[['Country', 'Leads', 'Evidence', 'Emerging Virality']])
        except Exception as e:
            st.error(f"Failed to load or process narrative file: {e}")
    else:
        st.info("Awaiting file upload for narrative analysis.")
