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
from io import StringIO, BytesIO # Import BytesIO for robust byte-level file reading


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
    elif "youtube.com" in url or "youtu.be" in url or "youtube.com" in url: # More robust YouTube detection
        return "YouTube"
    elif "instagram.com" in url:
        return "Instagram"
    elif "telegram.me" in url or "t.me" in url:
        return "Telegram"
    elif url.startswith("https://") or url.startswith("http://"):
        # Check if the URL contains common news/media domains, otherwise treat as general Media
        media_domains = ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "aljazeera.com", "lemonde.fr", "dw.com"]
        if any(domain in url for domain in media_domains):
            return "News/Media"
        return "Media"
    else:
        return "Unknown"

def extract_original_text(text):
    """
    Removes RT @user: or QT @user: prefixes (if any slipped through initial filter),
    @mentions, URLs, newlines, and normalizes spaces to get the core message
    for similarity analysis. This function now assumes the input `text` is
    already from a post that has been identified as 'original' (not a retweet/quoted tweet).
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Although posts starting with RT/QT are filtered out, a robust cleaner still
    # attempts to remove these and other common social media artifacts from the text content.
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()

    # Remove any remaining @mentions
    cleaned = re.sub(r'@\w+', '', cleaned).strip()

    # Remove URLs (http/https links)
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()

    # Remove newline/tab characters
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()

    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Lowercase the text for consistent comparison
    return cleaned.lower()

@st.cache_data(show_spinner="📥 Loading default datasets...")
def load_default_datasets():
    """Loads default Meltwater and CivicSignals datasets from GitHub."""
    base_url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/"
    urls = {
        "meltwater": f"{base_url}TogoJULYData%20-%20Sheet1.csv",
        "civicsignals": f"{base_url}togo-or-lome-or-togo-all-story-urls-20250707142808.csv"
    }

    meltwater_df = pd.DataFrame()
    civicsignals_df = pd.DataFrame()

    for key, url in urls.items():
        try:
            # Removed explicit encoding from default load as well for consistency
            df = pd.read_csv(url, sep=',')
            if not df.empty:
                if key == "meltwater":
                    meltwater_df = df
                elif key == "civicsignals":
                    civicsignals_df = df
                st.sidebar.success(f"✅ {key.capitalize()}: Data loaded from default URL")
            else:
                st.sidebar.warning(f"⚠️ {key.capitalize()}: Loaded but file is empty.")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to load {key}: {e}")

    return meltwater_df, civicsignals_df

# --- Combine Multiple Datasets ---
def combine_social_media_data(
    meltwater_df,
    civicsignals_df,
    openmeasure_df=None
):
    """
    Combines datasets from Meltwater, CivicSignals, and Open-Measure (optional) into a unified format
    with specific column mappings as per user's latest instructions.
    Handles case sensitivity of column names by converting to lowercase.
    """
    combined_dfs = []

    # Helper to safely get column by exact name, return NaN series if not found
    # Now expects column_name to be lowercase as df.columns will be lowercased
    def get_specific_col(df, col_name_lower):
        if col_name_lower in df.columns:
            return df[col_name_lower]
        return pd.Series([np.nan] * len(df), index=df.index)

    # Process Meltwater
    if meltwater_df is not None and not meltwater_df.empty:
        # Convert all column names to lowercase for robust lookup
        meltwater_df.columns = meltwater_df.columns.str.lower()
        mw = pd.DataFrame()
        mw['account_id'] = get_specific_col(meltwater_df, 'influencer') # Changed to lowercase
        mw['content_id'] = get_specific_col(meltwater_df, 'tweet id')   # Changed to lowercase
        mw['object_id'] = get_specific_col(meltwater_df, 'hit sentence')# Changed to lowercase
        mw['original_url'] = get_specific_col(meltwater_df, 'url')      # Changed to lowercase
        mw['timestamp_share'] = get_specific_col(meltwater_df, 'date')  # Changed to lowercase
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)

    # Process CivicSignals (Media Data)
    if civicsignals_df is not None and not civicsignals_df.empty:
        # Convert all column names to lowercase for robust lookup
        civicsignals_df.columns = civicsignals_df.columns.str.lower()
        cs = pd.DataFrame()
        cs['account_id'] = get_specific_col(civicsignals_df, 'media_name')   # Changed to lowercase
        cs['content_id'] = get_specific_col(civicsignals_df, 'stories_id')   # Changed to lowercase
        cs['object_id'] = get_specific_col(civicsignals_df, 'title')        # Changed to lowercase
        cs['original_url'] = get_specific_col(civicsignals_df, 'url')       # Changed to lowercase
        cs['timestamp_share'] = get_specific_col(civicsignals_df, 'publish_date') # Changed to lowercase
        cs['source_dataset'] = 'CivicSignals'
        combined_dfs.append(cs)

    # Process Open-Measure (optional)
    if openmeasure_df is not None and not openmeasure_df.empty:
        # Convert all column names to lowercase for robust lookup
        openmeasure_df.columns = openmeasure_df.columns.str.lower()
        om = pd.DataFrame()
        om['account_id'] = get_specific_col(openmeasure_df, 'actor_username') # Changed to lowercase
        om['content_id'] = get_specific_col(openmeasure_df, 'id')             # Changed to lowercase
        om['object_id'] = get_specific_col(openmeasure_df, 'text')            # Changed to lowercase
        om['original_url'] = get_specific_col(openmeasure_df, 'url')          # Changed to lowercase
        om['timestamp_share'] = get_specific_col(openmeasure_df, 'created_at')# Changed to lowercase
        om['source_dataset'] = 'OpenMeasure'
        combined_dfs.append(om)

    if not combined_dfs:
        return pd.DataFrame()

    combined = pd.concat(combined_dfs, ignore_index=True)

    # Clean and validate combined data
    # Ensure critical columns are not entirely NaN after specific mapping
    combined = combined.dropna(subset=['account_id', 'content_id', 'timestamp_share', 'object_id']).copy()

    combined['account_id'] = combined['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    combined['content_id'] = combined['content_id'].astype(str).str.replace('"', '', regex=False).str.strip() # Clean content_id
    combined['original_url'] = combined['original_url'].astype(str).replace('nan', '').fillna('')
    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('') # object_id should now contain text


    # Robust timestamp parsing
    date_formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', # ISO format with/without milliseconds
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p',
        '%A, %d %b %Y %H:%M:%S', '%b %d, %I:%M%p', '%d %b %Y %I:%M%p', # Added another format
        '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y',
    ]
    def parse_timestamp_robust(timestamp):
        if pd.isna(timestamp): return pd.NaT
        if isinstance(timestamp, (int, float)): return pd.NaT # Avoid parsing numbers as dates unless specific format

        # Try direct conversion first for common formats, then iterate
        parsed = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.notna(parsed): return parsed

        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce', utc=True)
                if pd.notna(parsed): return parsed
            except (ValueError, TypeError): continue
        return pd.NaT

    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)

    # 'object_id' is now the primary text column. Ensure it's string and fill NaNs.
    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')
    combined = combined[combined['object_id'].str.strip() != ""].copy() # Remove rows with empty text in object_id

    # Remove duplicates based on key fields
    combined = combined.drop_duplicates(subset=['account_id', 'content_id', 'object_id', 'timestamp_share']).reset_index(drop=True)

    return combined

# --- Final Preprocessing Function ---
def final_preprocess_and_map_columns(df):
    """
    Performs final preprocessing steps on the combined DataFrame:
    Renames 'original_url' to 'URL', cleans text, infers platform,
    and filters out retweets/quoted tweets for similarity analysis.
    Keeps all columns necessary for dashboard functionality.
    """
    if df.empty:
        return df

    df_processed = df.copy() # Work on a copy to avoid modifying original combined_df

    # 1. Rename 'original_url' to 'URL' only
    df_processed.rename(columns={
        'original_url': 'URL', # The URL is now this column
    }, inplace=True)

    # 2. Clean 'object_id' column (which holds the text content)
    df_processed['object_id'] = df_processed['object_id'].astype(str)
    df_processed = df_processed[df_processed['object_id'].notna()]
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].reset_index(drop=True)
    df_processed = df_processed[df_processed['object_id'].str.lower() != "nan"].reset_index(drop=True)

    def clean_text_for_display(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_processed['object_id'] = df_processed['object_id'].apply(clean_text_for_display)
    df_processed = df_processed[df_processed['object_id'].str.len() > 0].reset_index(drop=True)

    # NEW STEP: Filter out posts that are considered retweets or quoted tweets themselves
    # A post is considered a retweet/quoted tweet if its 'object_id' starts with "RT @" or "QT @"
    initial_rows_before_filter = len(df_processed)
    df_processed = df_processed[
        ~df_processed['object_id'].str.lower().str.startswith('rt @') &
        ~df_processed['object_id'].str.lower().str.startswith('qt @')
    ].reset_index(drop=True)

    # Notify user about filtered posts if any
    filtered_count = initial_rows_before_filter - len(df_processed)
    if filtered_count > 0:
        st.info(f"Filtered out {filtered_count} posts identified as retweets or quoted tweets for similarity analysis.")


    # 3. Ensure timestamp_share is datetime with UTC timezone
    df_processed['timestamp_share'] = pd.to_datetime(df_processed['timestamp_share'], errors='coerce', utc=True)
    df_processed = df_processed.dropna(subset=['timestamp_share']).reset_index(drop=True)

    # 4. Infer Platform from the 'URL' column
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)

    # 5. Extract original text (remove mentions/links specifically for similarity analysis, from object_id)
    # This column is crucial for similarity analysis and is separate from object_id
    df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)

    # Ensure 'Outlet' and 'Channel' columns exist for consistency if they were not explicitly mapped
    if 'Outlet' not in df_processed.columns:
        df_processed['Outlet'] = np.nan
    if 'Channel' not in df_processed.columns:
        df_processed['Channel'] = np.nan

    # Final check for empty DataFrame
    if df_processed.empty:
        st.error("❌ No valid data after complete preprocessing and filtering.")
        st.stop()

    # All columns needed for dashboard operations are retained in df_processed.
    return df_processed

# DBSCAN and Network Graph functions (Integrated)
def cluster_texts(df, eps=0.3, min_samples=2):
    """
    Clusters texts using DBSCAN based on TF-IDF vectorization and cosine similarity.
    Requires 'original_text' column.
    """
    if 'original_text' not in df.columns:
        st.error("Missing 'original_text' column for text clustering.")
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy

    texts_to_cluster = df['original_text'].astype(str).tolist()

    if not texts_to_cluster or all(text.strip() == "" for text in texts_to_cluster):
        st.warning("No valid text data for clustering. Assigning all to cluster -1 (noise).")
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts_to_cluster)
    except ValueError as e:
        st.warning(f"Could not create TF-IDF matrix for clustering: {e}. Assigning all to cluster -1 (noise).")
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy

    eps = max(0.01, min(0.99, eps))
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(tfidf_matrix)
    df_copy = df.copy()
    df_copy['cluster'] = clustering.labels_
    return df_copy

def build_user_interaction_graph(df, coordination_type="text"):
    """
    Builds a network graph of influencers based on shared coordination (text clusters or shared URLs).
    'coordination_type' can be 'text' or 'url'.
    """
    G = nx.Graph()

    # Use 'account_id' for influencers
    influencer_column = 'account_id'

    if coordination_type == "text":
        if 'cluster' not in df.columns:
            st.error("Missing 'cluster' column for text-based network graph.")
            return G, {}, {}

        grouped = df.groupby('cluster')
        for cluster_id, group in grouped:
            if cluster_id == -1 or len(group[influencer_column].unique()) < 2: # Exclude noise and single-influencer clusters
                # Add individual influencers to graph even if they are noise/isolated
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
            st.error("Missing 'URL' column for URL-based network graph.")
            return G, {}, {}

        # Group by URL to find shared links
        url_groups = df.groupby('URL')
        for url_shared, group in url_groups:
            if pd.isna(url_shared) or url_shared.strip() == "":
                continue # Skip empty/invalid URLs

            # Identify influencers who shared this URL
            users_sharing_url = group[influencer_column].dropna().unique().tolist()
            if len(users_sharing_url) < 2:
                # Add individual influencers even if they are isolated by URL
                for user in users_sharing_url:
                    if user not in G:
                        G.add_node(user, cluster=f"URL:{url_shared[:50]}...") # Provide a dummy cluster for coloring
                continue # Skip URLs shared by only one influencer for coordination edges

            # Add edges between all pairs of influencers who shared this URL
            for u1, u2 in combinations(users_sharing_url, 2):
                if G.has_edge(u1, u2):
                    G[u1][u2]['weight'] += 1
                else:
                    G.add_edge(u1, u2, weight=1)

    # Ensure all influencers from the filtered DataFrame are nodes, even if isolated
    all_influencers = df[influencer_column].dropna().unique().tolist()
    influencer_platform_map = df.groupby(influencer_column)['Platform'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').to_dict()

    for inf in all_influencers:
        if inf not in G.nodes():
            G.add_node(inf)
        G.nodes[inf]['platform'] = influencer_platform_map.get(inf, 'Unknown')

        # Assign cluster for coloring based on coordination_type
        if coordination_type == "text":
            influencer_clusters = df[df[influencer_column] == inf]['cluster'].dropna()
            if not influencer_clusters.empty:
                G.nodes[inf]['cluster'] = influencer_clusters.mode()[0]
            else:
                G.nodes[inf]['cluster'] = -2 # Special value for nodes not assigned to any cluster (or noise)
        elif coordination_type == "url":
            # For URL-based, assign a 'cluster' based on the first URL they shared or a general "shared URL" cluster
            shared_urls_for_inf = df[(df[influencer_column] == inf) & df['URL'].notna() & (df['URL'].str.strip() != '')]['URL'].unique()
            if len(shared_urls_for_inf) > 0:
                # Can assign a generic cluster for shared URLs or try to group by most common URL shared if desired
                G.nodes[inf]['cluster'] = f"SharedURL_Group_{hash(tuple(sorted(shared_urls_for_inf))) % 100}" # Simple way to group
            else:
                G.nodes[inf]['cluster'] = "NoSharedURL"


    pos = nx.spring_layout(G, seed=42, k=0.1, iterations=50)

    # The cluster_map for plotly should reflect the cluster attribute from G.nodes
    final_cluster_map = {node: G.nodes[node].get('cluster', -2) for node in G.nodes()}

    return G, pos, final_cluster_map

# Vectorized similarity function for text (unchanged, uses original_text)
def find_textual_similarities(df, threshold=0.85):
    """Finds pairs of text posts with high cosine similarity."""
    clean_df = df[['original_text', 'account_id', 'timestamp_share', 'Platform', 'URL']].copy() # Use account_id and timestamp_share
    clean_df['original_text'] = clean_df['original_text'].astype(str)
    clean_df = clean_df.dropna(subset=['original_text', 'account_id', 'timestamp_share'])
    clean_df = clean_df[clean_df['original_text'].str.strip() != ""].copy()
    texts = clean_df['original_text'].tolist()

    if len(texts) < 2:
        st.info("Not enough valid texts for similarity analysis.")
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as e:
        st.warning(f"Could not create TF-IDF matrix. Error: {e}. This might happen if all texts are very similar or empty after processing.")
        return pd.DataFrame()

    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0) # Set diagonal to 0 as a post is 100% similar to itself
    sim_matrix = np.triu(sim_matrix, k=1) # Only consider upper triangle to avoid duplicate pairs
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

        narrative_snippet = row1['original_text'][:150]
        if len(row1['original_text']) > 150:
            narrative_snippet += "..."
        if not narrative_snippet.strip():
            narrative_snippet = "Empty/Cleaned Text"

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
            'shared_narrative': narrative_snippet,
            'platforms_involved': f"{row1['Platform']},{row2['Platform']}"
        })
    return pd.DataFrame(similar_pairs)


# --- Cached Expensive Functions ---
@st.cache_data(show_spinner="🔍 Computing textual similarities...")
def cached_similarity_analysis(_df, threshold=0.85):
    return find_textual_similarities(_df, threshold)

@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df):
    """Performs text clustering using DBSCAN."""
    return cluster_texts(_df)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df_for_graph, coordination_type="text"):
    """Builds a user interaction network graph."""
    return build_user_interaction_graph(_df_for_graph, coordination_type)


# --- Sidebar: Data Source Selection ---
st.sidebar.header("📥 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ("Use Default Datasets", "Upload CSV Files")
)

combined_raw_df = pd.DataFrame()

if data_source == "Use Default Datasets":
    with st.spinner("📥 Loading and combining Meltwater and CivicSignal data..."):
        meltwater_df, civicsignals_df = load_default_datasets()
        combined_raw_df = combine_social_media_data(meltwater_df, civicsignals_df)
    if combined_raw_df.empty:
        st.warning("No data loaded from default datasets.")
        st.stop()
    st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from Meltwater and CivicSignal.")

elif data_source == "Upload CSV Files":
    st.sidebar.info("Upload CSVs from Meltwater, CivicSignal, and/or Open-Measure")

    uploaded_meltwater = st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater")
    uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv", "zip"], key="civicsignals")
    uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure")

    meltwater_df_upload = pd.DataFrame()
    civicsignals_df_upload = pd.DataFrame()
    openmeasure_df_upload = pd.DataFrame()

    if uploaded_meltwater:
        bytes_data = uploaded_meltwater.getvalue()
        try:
            # Removed encoding parameter, letting pandas infer, but kept sep=','
            meltwater_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',', low_memory=False)
            st.sidebar.success(f"✅ Meltwater CSV loaded successfully with auto-detected encoding (comma-separated).")
        except Exception as e:
            st.error(f"❌ Failed to read Meltwater CSV. Please ensure it is comma-separated: {e}")
            st.stop()


    if uploaded_civicsignals:
        if uploaded_civicsignals.type == "application/zip":
            st.sidebar.warning("Zip file uploaded for CivicSignals. Please ensure it contains a single CSV or extract manually.")
        else:
            bytes_data = uploaded_civicsignals.getvalue()
            try:
                # Removed encoding parameter, letting pandas infer, but kept sep=','
                civicsignals_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',')
                st.sidebar.success(f"✅ CivicSignals CSV loaded successfully with auto-detected encoding (comma-separated).")
            except Exception as e:
                st.error(f"❌ Failed to read CivicSignals CSV. Please ensure it is comma-separated: {e}")
                st.stop()

    if uploaded_openmeasure:
        bytes_data = uploaded_openmeasure.getvalue()
        try:
            # Removed encoding parameter, letting pandas infer, but kept sep=','
            openmeasure_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',')
            st.sidebar.success(f"✅ Open-Measure CSV loaded successfully with auto-detected encoding (comma-separated).")
        except Exception as e:
            st.error(f"❌ Failed to read Open-Measure CSV. Please ensure it is comma-separated: {e}")
            st.stop()


    if not meltwater_df_upload.empty or not civicsignals_df_upload.empty or not openmeasure_df_upload.empty:
        with st.spinner("Combining uploaded datasets..."):
            # Pass only the uploaded dataframes to ensure default data is not used
            combined_raw_df = combine_social_media_data(
                meltwater_df_upload if not meltwater_df_upload.empty else None,
                civicsignals_df_upload if not civicsignals_df_upload.empty else None,
                openmeasure_df_upload if not openmeasure_df_upload.empty else None
            )
        st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from uploaded files.")
    else:
        st.warning("Please upload at least one CSV file to proceed.")
        st.stop()

# --- Final Preprocess (after combining) ---
with st.spinner("⏳ Preprocessing and mapping combined data..."):
    df = final_preprocess_and_map_columns(combined_raw_df)

if df.empty:
    st.error("❌ No valid data after final preprocessing.")
    st.stop()

# --- Download Combined and Preprocessed Data ---
st.sidebar.markdown("### 💾 Download Combined & Preprocessed Data")
@st.cache_data
def convert_df_to_csv(data_frame):
    return data_frame.to_csv(index=False).encode('utf-8')

# Create a version of the DataFrame with only the requested core columns for download
download_df_columns = ['account_id', 'content_id', 'object_id', 'timestamp_share']
downloadable_df = df[download_df_columns].copy() if all(col in df.columns for col in download_df_columns) else pd.DataFrame()

if not downloadable_df.empty:
    combined_preprocessed_csv = convert_df_to_csv(downloadable_df)
    st.sidebar.download_button(
        "Download Preprocessed Dataset (Core Columns)",
        combined_preprocessed_csv,
        "preprocessed_combined_core_data.csv",
        "text/csv",
        help="Downloads the data after all preprocessing and column mapping, containing only account_id, content_id, object_id, and timestamp_share."
    )
else:
    st.sidebar.warning("Could not create downloadable dataset with core columns.")


# --- Sidebar Filters (Global Filters) ---
st.sidebar.header("🔍 Global Filters (Apply to all tabs)")

# Use 'timestamp_share' for filtering
if not pd.api.types.is_datetime64_any_dtype(df['timestamp_share']):
    st.error("timestamp_share column is not in datetime format after preprocessing. Cannot apply date filter.")
    st.stop()

min_date = df['timestamp_share'].min().date()
max_date = df['timestamp_share'].max().date()

selected_date_range = st.sidebar.date_input(
    "Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(selected_date_range) == 2:
    start_dt = pd.Timestamp(selected_date_range[0], tz='UTC')
    end_dt = pd.Timestamp(selected_date_range[1], tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
elif len(selected_date_range) == 1:
    start_dt = pd.Timestamp(selected_date_range[0], tz='UTC')
    end_dt = start_dt + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
else:
    start_dt = df['timestamp_share'].min()
    end_dt = df['timestamp_share'].max()

available_platforms_global = df['Platform'].dropna().astype(str).unique().tolist()
platforms_global = st.sidebar.multiselect(
    "Platforms",
    options=available_platforms_global,
    default=available_platforms_global
)

# Apply global filters using 'timestamp_share'
filtered_df_global = df[
    (df['timestamp_share'] >= start_dt) &
    (df['timestamp_share'] <= end_dt) &
    (df['Platform'].isin(platforms_global))
].copy()

if filtered_df_global.empty:
    st.warning("No data matches the selected global filters. Please adjust the date range or platforms.")
    st.stop()

# Export button for filtered data
st.sidebar.markdown("### 📄 Export Filtered Results")
filtered_csv_data = convert_df_to_csv(filtered_df_global)
st.sidebar.download_button("Download Filtered Data (All Columns)", filtered_csv_data, "filtered_dashboard_data.csv", "text/csv")


# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Summary Statistics")

    st.markdown("### 🔬 Preprocessed Data Sample")
    st.write("This table shows a small sample of the preprocessed data, displaying core identifiers for verification.")

    # Display only the requested core columns here
    display_cols_overview = ['account_id', 'content_id', 'object_id', 'timestamp_share']

    # Filter for columns that actually exist in the DataFrame
    existing_display_cols = [col for col in df.columns if col in display_cols_overview]

    st.dataframe(df[existing_display_cols].head(10))
    st.markdown("---")

    if not filtered_df_global.empty:
        st.write("This bar chart displays the top 10 influencers by their total number of posts within the filtered dataset.")
        # Use 'account_id' for influencers
        top_influencers = filtered_df_global['account_id'].value_counts().head(10)
        fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'account_id'})
        st.plotly_chart(fig_src, use_container_width=True)

        if 'Platform' in filtered_df_global.columns and not filtered_df_global['Platform'].empty:
            st.write("This bar chart illustrates the distribution of posts across various social media and news/media platforms.")
            all_platforms_counts = filtered_df_global['Platform'].value_counts()
            fig_platform = px.bar(all_platforms_counts, title="Post Distribution by Platform", labels={'value': 'Posts', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True)
        else:
            st.info("No 'Platform' column found or no data for platforms. This typically happens if no URLs are present in the data.")

        if 'Outlet' in filtered_df_global.columns and not filtered_df_global['Outlet'].empty and filtered_df_global['Outlet'].notna().any():
            st.write("This bar chart shows the top 10 media outlets or channels where content was published.")
            top_outlets = filtered_df_global['Outlet'].value_counts().head(10)
            fig_outlet = px.bar(top_outlets, title="Top 10 Media Outlets/Channels", labels={'value': 'Posts', 'index': 'Outlet'})
            st.plotly_chart(fig_outlet, use_container_width=True)
        elif 'Channel' in filtered_df_global.columns and not filtered_df_global['Channel'].empty and filtered_df_global['Channel'].notna().any():
            st.write("This bar chart illustrates the top 10 channels where content was published.")
            top_channels = filtered_df_global['Channel'].value_counts().head(10)
            fig_chan = px.bar(top_channels, title="Top 10 Channels", labels={'value': 'Posts', 'index': 'Channel'})
            st.plotly_chart(fig_chan, use_container_width=True)

        if 'Platform' in filtered_df_global.columns and not filtered_df_global['Platform'].empty:
            social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media'])].copy()

            if social_media_df.empty:
                st.info("Hashtag analysis skipped: No social media (non-'Media') content found in the filtered data.")
            else:
                # Use 'object_id' for text content
                if 'object_id' in social_media_df.columns and not social_media_df['object_id'].empty:
                    st.write("This bar chart highlights the top 10 most frequently used hashtags, focusing on social media content where hashtags are typically relevant.")
                    social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])

                    all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list if tags_list]

                    if all_hashtags:
                        hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                        fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                        st.plotly_chart(fig_ht, use_container_width=True)
                    else:
                        st.info("No hashtags found in the social media 'object_id' column.")
                else:
                    st.info("No 'object_id' column found or it's empty to extract hashtags from social media content.")
        else:
            st.info("Cannot determine platform for hashtag analysis (no 'Platform' column or empty).")


        st.write("This area chart visualizes the daily volume of posts over the selected date range.")
        # Use 'timestamp_share' for time series
        time_series = filtered_df_global.set_index('timestamp_share').resample('D').size()
        fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'timestamp_share': 'Date'})
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No data available to display summary statistics.")

# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🧠 Narrative Detection & Coordination")
    st.markdown("""
        This section helps identify **coordination** by finding very similar messages posted by different influencers.
        When different accounts share very similar messages, it can suggest they are working together or amplifying the same ideas.
        A high similarity score (close to 1.0) means the texts are almost identical.
        **Important:** Only original posts (not retweets or quoted tweets) are considered for this analysis to ensure meaningful similarity comparisons.
    """)

    st.markdown("---")
    st.subheader("Filters for Analysis")
    available_platforms_analysis = filtered_df_global['Platform'].dropna().astype(str).unique().tolist()
    platforms_analysis = st.multiselect(
        "Platforms to include in Similarity Analysis:",
        options=available_platforms_analysis,
        default=available_platforms_analysis,
        key="platforms_analysis_tab2"
    )

    analysis_df_filtered_by_platform = filtered_df_global[filtered_df_global['Platform'].isin(platforms_analysis)].copy()

    MAX_ROWS_SIMILARITY = st.slider("Max posts to analyze for similarity (for performance)", 100, 1000, 300, key="max_rows_similarity")

    # Ensure original_text column exists and use .str.strip()
    analysis_df = analysis_df_filtered_by_platform[
        (analysis_df_filtered_by_platform['original_text'].notna()) &
        (analysis_df_filtered_by_platform['original_text'].astype(str).str.strip() != "")
    ].head(MAX_ROWS_SIMILARITY).copy()

    if analysis_df.empty:
        st.info("No valid text data available for similarity analysis after applying filters and row limit.")
    else:
        with st.spinner(f"🔍 Finding coordinated narratives among {len(analysis_df)} original posts..."):
            sim_df = cached_similarity_analysis(analysis_df, threshold=0.85)

        if not sim_df.empty:
            st.success(f"✅ Found {len(sim_df)} similar pairs between original posts.")
            narrative_summary = sim_df.groupby('shared_narrative').agg(
                share_count=('similarity', 'count'),
                account_ids_involved=('account_id1', lambda x: ", ".join(x.astype(str).unique()[:5]) + ("..." if len(x.unique()) > 5 else "")),
                platforms_involved=('platforms_involved', lambda x: ", ".join(sorted(list(set([p.strip() for sublist in x.tolist() for p in sublist.split(',') if p.strip() != ""])))))
            ).sort_values(by='share_count', ascending=False).reset_index()

            st.markdown("### 🔝 Top Coordinated Narratives")
            st.write("This bar chart shows the top 10 narrative snippets that are shared across multiple posts, indicating potential coordination.")
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

            st.write("This table summarizes the top coordinated narratives, including the number of shares, involved influencers, and the platforms they appeared on.")
            st.dataframe(narrative_summary)

            st.markdown("### 🔄 Full Similarity Pairs (Original Posts Only)")
            st.write("This table lists all detected pairs of similar texts between *original posts*, along with their influencers, platforms, timestamps, similarity scores, and links to the original posts for verification.")

            display_sim_df = sim_df[['text1', 'account_id1', 'platform1', 'timestamp_share1', 'url1', 'text2', 'account_id2', 'platform2', 'timestamp_share2', 'url2', 'similarity']].copy()
            display_sim_df['url1'] = display_sim_df['url1'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) and x.startswith('http') else '')
            display_sim_df['url2'] = display_sim_df['url2'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) and x.startswith('http') else '')

            st.markdown(display_sim_df.to_html(escape=False), unsafe_allow_html=True)

        else:
            st.info("No significant similarities found above threshold between original posts.")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")

    st.markdown("---")
    st.subheader("Filters for Analysis)")
    available_platforms_network = filtered_df_global['Platform'].dropna().astype(str).unique().tolist()
    platforms_network = st.multiselect(
        "Platforms to include in Network & Risk Analysis:",
        options=available_platforms_network,
        default=available_platforms_network,
        key="platforms_network_tab3"
    )

    network_df_filtered_by_platform = filtered_df_global[filtered_df_global['Platform'].isin(platforms_network)].copy()

    coordination_basis = st.radio(
        "Choose basis for Coordination and Network Analysis:",
        ("Text Content (Narrative Similarity)", "Shared URLs"),
        key="coordination_basis_selector"
    )

    if coordination_basis == "Text Content (Narrative Similarity)":
        try:
            # Ensure original_text column exists and use .str.strip()
            df_for_clustering = network_df_filtered_by_platform[
                (network_df_filtered_by_platform['original_text'].notna()) &
                (network_df_filtered_by_platform['original_text'].astype(str).str.strip() != "")
            ].copy()

            if df_for_clustering.empty:
                st.info("No valid text data for clustering analysis.")
                clustered_df = pd.DataFrame()
            else:
                clustered_df = cached_clustering(df_for_clustering)

            if not clustered_df.empty and 'cluster' in clustered_df.columns:
                cluster_counts = clustered_df['cluster'].value_counts()
                if -1 in cluster_counts.index:
                    noise_count = cluster_counts[-1]
                    cluster_counts = cluster_counts.drop(index=-1)
                    st.info(f"💡 {noise_count} posts were identified as noise (Cluster -1) and are excluded from cluster visualization but still included in the network graph if they are influencers.")

                if not cluster_counts.empty:
                    st.markdown("### 🤖 Detected Coordination Clusters (Text-based)")
                    st.write("This bar chart visualizes the sizes of detected clusters, where each cluster represents a group of coordinated texts identified by their similarity.")
                    fig_clust = px.bar(
                        cluster_counts,
                        title="Cluster Sizes",
                        labels={'value': 'Member Count', 'index': 'Cluster ID'},
                        color=cluster_counts.index.astype(str),
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig_clust, use_container_width=True)

                    st.markdown("#### Cluster Summary")
                    st.write("This table provides a summary of each detected text-based cluster, including the number of posts, unique influencers, example influencers, and platforms involved.")
                    if not clustered_df[clustered_df['cluster'] != -1].empty:
                        cluster_details = clustered_df[clustered_df['cluster'] != -1].groupby('cluster').agg(
                            num_posts=('object_id', 'count'),
                            num_influencers=('account_id', 'nunique'),
                            example_influencers=('account_id', lambda x: ", ".join(x.unique()[:3]) + ("..." if len(x.unique()) > 3 else "")),
                            platforms=('Platform', lambda x: ", ".join(sorted(x.unique())))
                        ).sort_values(by='num_posts', ascending=False).reset_index()
                        st.dataframe(cluster_details)
                    else:
                        st.info("No detailed summary for text-based clusters (all posts might be noise or too few posts for coordination).")

                    st.write("This table shows the influencers, their text posts, timestamps, and their assigned text-based cluster IDs.")
                    st.dataframe(clustered_df[['account_id', 'Platform', 'object_id', 'timestamp_share', 'cluster']])
                else:
                    st.info("No significant text-based clusters detected (all posts might be noise or too few posts for clustering).")
            else:
                st.info("No data available for text-based clustering.")

        except Exception as e:
            st.warning(f"⚠️ Text-based clustering analysis failed: {e}")

    elif coordination_basis == "Shared URLs":
        st.markdown("### 🔗 Coordination by Shared URLs")
        st.write("This section identifies coordination where multiple influencers share the exact same external URL.")

        url_coordination_df = network_df_filtered_by_platform[
            (network_df_filtered_by_platform['URL'].notna()) &
            (network_df_filtered_by_platform['URL'].str.strip() != '') &
            (network_df_filtered_by_platform['URL'] != 'Unknown')
        ].copy()

        if url_coordination_df.empty:
            st.info("No valid URLs found for shared URL analysis in the filtered data.")
        else:
            # Group by URL and count unique influencers per URL
            url_summary = url_coordination_df.groupby('URL').agg(
                num_posts=('content_id', 'count'),
                num_influencers=('account_id', 'nunique'),
                influencers=('account_id', lambda x: ", ".join(x.unique()[:5]) + ("..." if len(x.unique()) > 5 else "")),
                platforms=('Platform', lambda x: ", ".join(sorted(x.unique())))
            ).reset_index()

            # Filter for URLs shared by 2 or more influencers
            shared_by_multiple_influencers = url_summary[url_summary['num_influencers'] >= 2]

            if not shared_by_multiple_influencers.empty:
                st.write("This table lists URLs that have been shared by two or more unique influencers, indicating potential coordination through link sharing.")
                st.dataframe(shared_by_multiple_influencers.sort_values(by='num_influencers', ascending=False))

                st.write("This bar chart highlights the top 10 URLs shared by the most unique influencers, signifying popular shared content within coordinated efforts.")
                fig_url_shares = px.bar(
                    shared_by_multiple_influencers.nlargest(10, 'num_influencers'),
                    x='num_influencers',
                    y='URL',
                    orientation='h',
                    title="Top 10 URLs Shared by Multiple Influencers",
                    labels={'num_influencers': 'Number of Influencers', 'URL': 'Shared URL'},
                    color='num_influencers',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_url_shares, use_container_width=True)
            else:
                st.info("No URLs found that are shared by 2 or more unique influencers in the filtered data.")

    st.markdown("---")
    max_influencers_graph = st.slider(
        "Max Influencers for Network Graph (for performance)",
        min_value=10, max_value=200, value=50, step=10,
        help="Limit the number of influencers displayed in the network graph to improve performance.",
        key="max_influencers_graph"
    )

    num_top_clusters_to_show = 0
    if coordination_basis == "Text Content (Narrative Similarity)":
        if 'clustered_df' in locals() and not clustered_df.empty and 'cluster' in clustered_df.columns:
            cluster_counts_for_slider = clustered_df['cluster'].value_counts()
            if -1 in cluster_counts_for_slider.index:
                cluster_counts_for_slider = cluster_counts_for_slider.drop(index=-1)

            if not cluster_counts_for_slider.empty:
                num_top_clusters_to_show = st.slider(
                    "Display Top N Largest Text Clusters in Network Graph",
                    min_value=1, max_value=len(cluster_counts_for_slider.index), value=min(5, len(cluster_counts_for_slider.index)), step=1,
                    help="Select how many of the largest coordinated text clusters to visualize in the network graph.",
                    key='num_top_clusters_to_show'
                )

    try:
        if coordination_basis == "Text Content (Narrative Similarity)":
            graph_df_base = clustered_df.copy() if 'clustered_df' in locals() and not clustered_df.empty else network_df_filtered_by_platform.copy()
            if graph_df_base.empty or graph_df_base['account_id'].dropna().empty:
                st.info("No valid influencer data to build the network graph for text-based coordination.")
                G, pos, cluster_map = nx.Graph(), {}, {}
            else:
                if num_top_clusters_to_show > 0 and 'cluster_counts_for_slider' in locals() and not cluster_counts_for_slider.empty:
                    top_cluster_ids = cluster_counts_for_slider.nlargest(num_top_clusters_to_show).index.tolist()
                    influencers_in_top_clusters = graph_df_base[graph_df_base['cluster'].isin(top_cluster_ids)]['account_id'].unique().tolist()
                    graph_df_filtered_for_top_clusters = graph_df_base[graph_df_base['account_id'].isin(influencers_in_top_clusters)].copy()
                else:
                    graph_df_filtered_for_top_clusters = graph_df_base.copy()

                top_active_influencers_in_subset = graph_df_filtered_for_top_clusters['account_id'].value_counts().nlargest(max_influencers_graph).index.tolist()
                graph_df_subset = graph_df_filtered_for_top_clusters[graph_df_filtered_for_top_clusters['account_id'].isin(top_active_influencers_in_subset)].copy()

                # Check if graph_df_subset is empty after all filters
                if graph_df_subset.empty:
                    st.info("No data available to build the network graph after applying all text-based filters and limits.")
                    G, pos, cluster_map = nx.Graph(), {}, {}
                else:
                    G, pos, cluster_map = cached_network_graph(graph_df_subset, coordination_type="text")

        elif coordination_basis == "Shared URLs":
            # For URL-based, use the network_df_filtered_by_platform directly
            # Ensure URLs are not empty/nan before passing
            graph_df_base = network_df_filtered_by_platform[
                (network_df_filtered_by_platform['URL'].notna()) &
                (network_df_filtered_by_platform['URL'].str.strip() != '') &
                (network_df_filtered_by_platform['URL'] != 'Unknown')
            ].copy()

            if graph_df_base.empty or graph_df_base['account_id'].dropna().empty:
                st.info("No valid URL data or influencer data to build the network graph for URL-based coordination.")
                G, pos, cluster_map = nx.Graph(), {}, {}
            else:
                top_active_influencers_in_subset = graph_df_base['account_id'].value_counts().nlargest(max_influencers_graph).index.tolist()
                graph_df_subset = graph_df_base[graph_df_base['account_id'].isin(top_active_influencers_in_subset)].copy()

                # Check if graph_df_subset is empty after all filters
                if graph_df_subset.empty:
                    st.info("No data available to build the network graph after applying all URL-based filters and limits.")
                    G, pos, cluster_map = nx.Graph(), {}, {}
                else:
                    G, pos, cluster_map = cached_network_graph(graph_df_subset, coordination_type="url")

        if not G.nodes():
            st.info("No nodes to display in the network graph. This might be due to filtered data or issues in graph creation.")
        else:
            st.markdown("### 🕸️ User Interaction Network")
            st.markdown("""
                This interactive graph shows how different **influencers** (represented by **nodes** or circles) are connected.
                A line (or **edge**) between two influencers means they have been identified as coordinating based on the selected method (either sharing very similar text content or sharing the exact same URLs).

                **How to interpret the colors:**
                The colors of the nodes are assigned automatically to visually group influencers that belong to the same detected cluster (for text-based coordination) or a similar group (for URL-based coordination).
                For example, all influencers within the 'blue' group are part of one coordinated cluster, while those in the 'green' group belong to a different one.
                The specific meaning of each color is not fixed (e.g., 'red' doesn't always mean the same thing across different analyses), but its purpose is to help you quickly see which influencers are working together on similar themes or sharing similar links.
            """)

            edge_trace = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'))

            node_colors_raw = [G.nodes[node].get('cluster', 'N/A') for node in G.nodes()]
            unique_clusters = sorted(list(set(node_colors_raw)))

            color_palette = px.colors.qualitative.Set3 + px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
            extended_color_palette = color_palette * ((len(unique_clusters) // len(color_palette)) + 1)

            color_map_for_plot = {c_id: extended_color_palette[i % len(extended_color_palette)] for i, c_id in enumerate(unique_clusters)}
            node_color_vals = [color_map_for_plot[c] for c in node_colors_raw]

            influencer_post_counts = graph_df_subset['account_id'].value_counts()
            max_node_size_display = 25
            min_node_size_display = 10

            if not influencer_post_counts.empty:
                min_posts_val = influencer_post_counts.min()
                max_posts_val = influencer_post_counts.max()
                if max_posts_val == min_posts_val:
                    node_sizes_raw = [min_node_size_display + (max_node_size_display - min_node_size_display)/2] * len(G.nodes())
                else:
                    node_sizes_raw = [
                        min_node_size_display + (max_node_size_display - min_node_size_display) * ((influencer_post_counts.get(node, min_posts_val) - min_posts_val) / (max_posts_val - min_posts_val))
                        for node in G.nodes()
                    ]
            else:
                node_sizes_raw = [min_node_size_display] * len(G.nodes())

            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                text=[f"Influencer: {node}<br>Posts in Graph: {influencer_post_counts.get(node, 0)}<br>Coordination Group: {G.nodes[node].get('cluster', 'N/A')}<br>Platform: {G.nodes[node].get('platform', 'N/A')}" for node in G.nodes()],
                mode='markers+text',
                textposition="top center",
                marker=dict(
                    size=node_sizes_raw,
                    color=node_color_vals,
                    line=dict(width=2, color='darkblue'),
                ),
                hoverinfo='text'
            )

            fig_net = go.Figure(data=edge_trace + [node_trace],
                                layout=go.Layout(
                                    title="Influencer Coordination Network (Click & Drag to Explore)",
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20, l=5, r=5, t=60),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    height=600))
            st.plotly_chart(fig_net, use_container_width=True)

            if unique_clusters:
                st.markdown("#### Coordination Group Color Legend:")
                legend_items = []
                for i, cluster_id in enumerate(unique_clusters):
                    label = f"Group: {cluster_id}"
                    color_hex = color_map_for_plot[cluster_id]
                    legend_items.append(f"<span style='color:{color_hex}'>●</span> {label}")
                st.markdown("<br>".join(legend_items), unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"⚠️ Network graph failed: {e}")

    st.markdown("### ⚠️ High-Risk Influencers")
    st.markdown("""
        **High-risk influencers** are those who frequently participate in **coordination**.
        This chart highlights influencers who appear in 3 or more **similar messages** (from the Similarity Analysis section), where only *original posts* are considered.
        A high count here could indicate that an influencer is a central figure in spreading specific narratives or is part of a concentrated effort.
    """)
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
                    title="Influencers in ≥3 Coordinated Messages (Original Posts Only)",
                    labels={'value': 'Coordination Instances', 'index': 'account_id'},
                    color='value',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_hr, use_container_width=True)
            else:
                st.info("No influencers found participating in 3 or more coordinated messages from original posts.")
        else:
            st.info("No coordinated narratives detected from original posts to identify high-risk influencers.")
    except Exception as e:
        st.warning(f"Risk analysis failed: {e}")
