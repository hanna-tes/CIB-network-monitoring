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
def cluster_texts(df, eps, min_samples, max_features):
    if 'original_text' not in df.columns or df['original_text'].nunique() <= 1:
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy
    
    texts_to_cluster = df['original_text'].astype(str).tolist()
    if not texts_to_cluster or all(t.strip() == "" for t in texts_to_cluster):
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
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

def find_coordinated_groups(df, threshold, max_features):
    """
    Groups highly similar posts into coordination groups for better analysis.
    Crucially, a group is only considered coordinated if it involves more than one unique account.
    """
    text_col = 'original_text'
    social_media_platforms = {'TikTok', 'Facebook', 'X', 'YouTube', 'Instagram', 'Telegram'}
    
    coordination_groups = {}
    
    clustered_groups = df[df['cluster'] != -1].groupby('cluster')
    
    for cluster_id, group in clustered_groups:
        if len(group) < 2:
            continue
            
        # Use TF-IDF for similarity within the small cluster
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col]].copy()
        clean_df = clean_df.rename(columns={text_col: 'text', 'timestamp_share': 'Timestamp'})
        clean_df = clean_df.reset_index(drop=True)
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(3, 5),
            max_features=max_features
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_df['text'])
        except Exception:
            continue
        
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        # Build an adjacency list for connected components
        adj = {i: [] for i in range(len(clean_df))}
        for i in range(len(clean_df)):
            for j in range(i + 1, len(clean_df)):
                if cosine_sim[i, j] >= threshold:
                    adj[i].append(j)
                    adj[j].append(i)
                    
        visited = set()
        group_id_counter = 0
        
        for i in range(len(clean_df)):
            if i not in visited:
                group_indices = []
                q = [i]
                visited.add(i)
                while q:
                    u = q.pop(0)
                    group_indices.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                
                if len(group_indices) > 1:
                    # Collect all posts in this connected component
                    group_posts = clean_df.iloc[group_indices].copy()
                    
                    # --- CORE LOGIC FOR AMPLIFICATION: Only consider a group coordinated if there are multiple unique accounts ---
                    if len(group_posts['account_id'].unique()) > 1:
                        # Determine a single representative snippet for the group
                        representative_text = group_posts['text'].iloc[0]
                        snippet = representative_text[:120] + ("..." if len(representative_text) > 120 else "")

                        # Calculate max similarity in the group (for a score)
                        group_sim_scores = cosine_sim[np.ix_(group_indices, group_indices)]
                        max_sim = group_sim_scores.max() if group_sim_scores.size > 0 else 0.0

                        # Assign a unique ID and store the group data
                        coordination_groups[f"group_{group_id_counter}"] = {
                            "posts": group_posts.to_dict('records'),
                            "num_posts": len(group_posts),
                            "num_accounts": len(group_posts['account_id'].unique()),
                            "shared_narrative_snippet": snippet,
                            "max_similarity_score": round(max_sim, 3),
                            "coordination_type": "TBD" # Will be set below
                        }
                        group_id_counter += 1

    # Now, process groups to determine coordination type
    final_groups = []
    for group_id, group_data in coordination_groups.items():
        posts_df = pd.DataFrame(group_data['posts'])
        platforms = posts_df['Platform'].unique()
        
        social_media_platforms_in_group = [p for p in platforms if p in social_media_platforms]
        media_platforms_in_group = [p for p in platforms if p in {'News/Media', 'Media'}]

        if len(media_platforms_in_group) > 1 and len(social_media_platforms_in_group) == 0:
            coordination_type = "Syndication (Media Outlets)"
        elif len(social_media_platforms_in_group) > 1 and len(media_platforms_in_group) == 0:
            coordination_type = "Coordinated Amplification (Social Media)"
        elif len(social_media_platforms_in_group) > 0 and len(media_platforms_in_group) > 0:
            coordination_type = "Media-to-Social Replication"
        else:
            coordination_type = "Other / Uncategorized"
        
        group_data['coordination_type'] = coordination_type
        final_groups.append(group_data)
        
    return final_groups


def build_user_interaction_graph(df, coordination_type="text"):
    G = nx.Graph()
    influencer_column = 'account_id'

    if coordination_type == "text":
        if 'cluster' not in df.columns:
            return G, {}, {}
        grouped = df.groupby('cluster')
        for cluster_id, group in grouped:
            # Only add edges if the cluster involves more than one unique account
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
            # Only add edges if the URL is shared by more than one unique account
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
            # Assign cluster if available, otherwise -2 for no coordination
            G.nodes[inf]['cluster'] = clusters.mode()[0] if not clusters.empty else -2
        elif coordination_type == "url":
            shared_urls = df[(df[influencer_column] == inf) & df['URL'].notna() & (df['URL'].str.strip() != '')]['URL'].unique()
            G.nodes[inf]['cluster'] = f"SharedURL_Group_{hash(tuple(sorted(shared_urls))) % 100}" if len(shared_urls) > 0 else "NoSharedURL"

    # --- New Logic: Filter nodes by degree centrality before layout ---
    if G.nodes():
        node_degrees = dict(G.degree())
        sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)
        top_n_nodes = sorted_nodes[:st.session_state.max_nodes_to_display]
        subgraph = G.subgraph(top_n_nodes)
        
        # Recalculate layout on the smaller subgraph
        pos = nx.spring_layout(subgraph, seed=42, k=0.1, iterations=50)
        cluster_map = {node: G.nodes[node].get('cluster', -2) for node in subgraph.nodes()}
        return subgraph, pos, cluster_map
    else:
        return G, {}, {}


# --- Cached Functions ---
@st.cache_data(show_spinner="🔍 Finding coordinated posts within clusters...")
def cached_find_coordinated_groups(_df, threshold, max_features, data_source="default"):
    return find_coordinated_groups(_df, threshold, max_features)

@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df, eps, min_samples, max_features, data_source="default"):
    return cluster_texts(_df, eps, min_samples, max_features)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df_for_graph, coordination_type="text", data_source="default"):
    # This function is now a proxy and will call the main build_user_interaction_graph
    # which will use the session state to get the max_nodes_to_display
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

    # Handle file uploads
    meltwater_df_upload = pd.DataFrame()
    civicsignals_df_upload = pd.DataFrame()
    openmeasure_df_upload = pd.DataFrame()

    def read_uploaded_file(uploaded_file, file_name):
        if not uploaded_file:
            return pd.DataFrame()
        
        bytes_data = uploaded_file.getvalue()
        encodings = ['utf-8-sig', 'utf-16le', 'utf-16be', 'utf-16', 'latin1', 'cp1252']
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

        sample_line = decoded_content.strip().splitlines()[0]
        sep = '\t' if '\t' in sample_line else ','
        
        try:
            df = pd.read_csv(StringIO(decoded_content), sep=sep, low_memory=False)
            st.sidebar.success(f"✅ {file_name}: Loaded {len(df)} rows (sep='{sep}', enc='{detected_enc}')")
            return df
        except Exception as e:
            st.error(f"❌ Failed to parse {file_name} CSV after decoding: {e}")
            return pd.DataFrame()

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

    if combined_raw_df.empty:
        st.warning("No data loaded from uploaded files.")
        st.stop()
    st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from uploaded datasets.")
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

# Add new control to limit posts
st.sidebar.markdown("---")
st.sidebar.subheader("⏩ Performance Controls")
max_posts_for_analysis = st.sidebar.number_input(
    "Limit Posts for Analysis (0 for all)",
    min_value=0,
    value=0,
    step=1000,
    help="To speed up analysis on large datasets, enter a number to process a random sample of posts. Set to 0 to use all posts."
)
st.sidebar.markdown(f"**Filtered Posts:** `{len(filtered_df_global):,}`")

# Apply sampling if requested
if max_posts_for_analysis > 0 and len(filtered_df_global) > max_posts_for_analysis:
    df_for_analysis = filtered_df_global.sample(n=max_posts_for_analysis, random_state=42).copy()
    st.sidebar.warning(f"⚠️ Analyzing a random sample of **{len(df_for_analysis):,}** posts to improve performance.")
else:
    df_for_analysis = filtered_df_global.copy()
    st.sidebar.info(f"✅ Analyzing all **{len(df_for_analysis):,}** posts.")


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
    st.subheader("🕵️‍♀️ Similarity & Coordination Analysis")
    st.markdown("""
        This section identifies coordinated activities by analyzing posts with high textual similarity.
    """)
    
    # --- Repost/Replication Count Table (New Section) ---
    st.subheader("🔄 Repost & Replication Count")
    st.markdown("This table shows content that has been reposted or replicated across different accounts, providing a direct count of identical posts.")

    if coordination_mode == "Text Content":
        with st.spinner("🔢 Counting reposts and replications..."):
            repost_counts = df_for_analysis.groupby('original_text').filter(lambda x: len(x) > 1).groupby('original_text').agg(
                repost_count=('account_id', 'size'),
                first_account_id=('account_id', 'first'),
                first_platform=('Platform', 'first'),
                first_timestamp=('timestamp_share', 'first'),
                first_url=('URL', 'first')
            ).reset_index()

            if not repost_counts.empty:
                repost_counts = repost_counts.rename(columns={'original_text': 'original_content'})
                repost_counts['first_timestamp'] = pd.to_datetime(repost_counts['first_timestamp'], unit='s', utc=True)
                repost_counts = repost_counts.sort_values('repost_count', ascending=False)
                
                display_cols = ['repost_count', 'original_content', 'first_account_id', 'first_platform', 'first_timestamp', 'first_url']
                st.info(f"✅ Found {len(repost_counts)} unique pieces of content with 2 or more reposts.")
                st.dataframe(repost_counts[display_cols].head(20), height=500, use_container_width=True)
                
                repost_counts_csv = convert_df_to_csv(repost_counts)
                st.download_button(
                    "Download Repost Count CSV",
                    repost_counts_csv,
                    "repost_counts.csv",
                    "text/csv",
                    help="Downloads the list of content and their repost counts."
                )
            else:
                st.info("No exact reposts or replications found in the selected data.")
    else:
        st.info("Repost count is only available when 'Text Content' is selected as the coordination mode.")

    st.markdown("---") # Separator for the next section

    if coordination_mode == "Text Content":
        st.subheader("Textual Similarity Analysis")
        st.markdown("""
            - **Syndication**: When a content shared by one media outlet is replicated by other media outlets word-for-word.
            - **Coordinated Amplification**: When highly similar posts are shared by different accounts on social media platforms.
            - **Media-to-Social Replication**: When content from a media outlet is replicated on a social media platform.
        """)

        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Similarity Tuning (Advanced)")
        eps = st.sidebar.slider(
            "DBSCAN Cluster Similarity Threshold (eps)",
            min_value=0.1, max_value=1.0, value=0.3, step=0.05,
            help="Lower values create more, smaller clusters. Higher values create fewer, larger clusters. A lower value can speed up the pairwise comparison stage."
        )
        min_samples = st.sidebar.slider(
            "DBSCAN Minimum Samples per Cluster",
            min_value=2, max_value=10, value=2, step=1,
            help="Minimum number of posts to form a cluster. Increasing this can make clustering faster by ignoring small groups."
        )
        max_features = st.sidebar.slider(
            "TF-IDF Max Features",
            min_value=1000, max_value=10000, value=5000, step=1000,
            help="The maximum number of unique words to consider. Lowering this can significantly speed up vectorization."
        )

        threshold_sim = st.slider("Similarity Threshold for Pairs", min_value=0.75, max_value=0.99, value=0.90, step=0.01)
        
        # New approach: First cluster the data, then find similarities within clusters
        with st.spinner("🚀 Speeding up analysis with clustering..."):
            clustered_df = cached_clustering(df_for_analysis, eps=eps, min_samples=min_samples, max_features=max_features, data_source=data_source + "_" + coordination_mode)
            
        with st.spinner("🕵️‍♂️ Finding similar posts within clusters..."):
            coordinated_groups = cached_find_coordinated_groups(clustered_df, threshold=threshold_sim, max_features=max_features, data_source=data_source + "_" + coordination_mode + "_" + str(threshold_sim))

        if coordinated_groups:
            st.info(f"✅ Found {len(coordinated_groups)} groups of posts with similarity score ≥ {threshold_sim:.2f}.")

            st.markdown("### Coordinated Amplification & Syndication Results")
            
            for i, group in enumerate(coordinated_groups):
                st.markdown(f"#### Group {i+1}: {group['coordination_type']}")
                st.write(f"**Shared Narrative:** {group['shared_narrative_snippet']}")
                st.write(f"**Number of Posts:** {group['num_posts']} | **Number of Unique Accounts:** {group['num_accounts']} | **Max Similarity:** {group['max_similarity_score']}")
                
                posts_df = pd.DataFrame(group['posts'])
                # Convert the existing 'Timestamp' column to datetime and rename for display.
                posts_df['Timestamp'] = pd.to_datetime(posts_df['Timestamp'], unit='s', utc=True)
                posts_df = posts_df.rename(columns={'account_id': 'Account ID', 'Platform': 'Platform', 'URL': 'URL'})
                posts_df = posts_df[['Account ID', 'Platform', 'Timestamp', 'URL']]
                st.dataframe(posts_df, use_container_width=True)
                st.markdown("---")

            # --- Download button for similar pairs ---
            flat_groups = [
                {
                    'group_id': i,
                    'coordination_type': g['coordination_type'],
                    'shared_narrative_snippet': g['shared_narrative_snippet'],
                    'max_similarity_score': g['max_similarity_score'],
                    **p
                } for i, g in enumerate(coordinated_groups) for p in g['posts']
            ]
            similar_groups_df = pd.DataFrame(flat_groups)
            # Ensure a clean DataFrame before writing to CSV to avoid errors.
            similar_groups_df.rename(columns={
                'account_id': 'Account ID',
                'Platform': 'Platform',
                'Timestamp': 'Timestamp (s)',
                'URL': 'URL'
            }, inplace=True)
            similar_groups_csv = convert_df_to_csv(similar_groups_df)
            st.download_button(
                "Download Coordinated Groups CSV",
                similar_groups_csv,
                f"coordinated_groups_analysis_{threshold_sim}.csv",
                "text/csv",
                help="Downloads the table of identified coordinated post groups."
            )

        else:
            st.warning("No significant textual similarities found above the selected threshold.")

    elif coordination_mode == "Shared URLs":
        st.subheader("Shared URLs Analysis")
        with st.spinner("🔗 Finding posts that share the same URLs..."):
            shared_url_df = df_for_analysis[  # Use df_for_analysis
                df_for_analysis['URL'].notna() &
                (df_for_analysis['URL'].str.strip() != "")
            ].copy()
            if not shared_url_df.empty:
                url_counts = shared_url_df['URL'].value_counts()
                coordination_urls = url_counts[url_counts > 1].index.tolist()
                coordination_df = shared_url_df[shared_url_df['URL'].isin(coordination_urls)].sort_values(by='URL')
                
                if not coordination_df.empty:
                    st.info(f"✅ Found {len(coordination_urls)} URLs shared by more than one account.")
                    st.markdown("### Posts Sharing the Same URL")
                    st.dataframe(coordination_df[['account_id', 'Platform', 'timestamp_share', 'URL']].reset_index(drop=True), height=500, use_container_width=True)
                    
                    # --- Download button for shared URLs ---
                    shared_url_csv = convert_df_to_csv(coordination_df)
                    st.download_button(
                        "Download Shared URLs CSV",
                        shared_url_csv,
                        "shared_urls_analysis.csv",
                        "text/csv",
                        help="Downloads the list of posts that share the same URLs."
                    )
                else:
                    st.warning("No URLs were shared by more than one account.")
            else:
                st.warning("No valid URLs found in the dataset.")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🕸️ Network Graph of Coordinated Activity")

    # --- New Slider for Node Limiting ---
    st.markdown("Use the slider below to limit the number of accounts displayed in the network graph.")
    if 'max_nodes_to_display' not in st.session_state:
        st.session_state.max_nodes_to_display = 40  # Default value
    st.session_state.max_nodes_to_display = st.slider(
        "Maximum Nodes to Display in Graph",
        min_value=10, max_value=200, value=st.session_state.max_nodes_to_display, step=10,
        help="Limit the graph to the top N most central accounts to improve visibility and focus on key influencers."
    )
    st.markdown("---") # Separator

    st.markdown("This visualization shows a network of accounts involved in coordinated activity. A link between two accounts means they posted similar content or shared the same URL.")
    
    # Decide which DataFrame to use for the graph based on coordination mode
    # Use df_for_analysis for both network graph modes
    if coordination_mode == "Text Content":
        df_for_graph = df_for_analysis
        with st.spinner("🗂️ Pre-processing data for network graph..."):
            clustered_df_for_graph = cached_clustering(df_for_graph, eps=eps, min_samples=min_samples, max_features=max_features, data_source=data_source + "_" + coordination_mode)
        
        G_text, pos_text, cluster_map_text = cached_network_graph(clustered_df_for_graph, coordination_type="text", data_source=data_source + "_" + coordination_mode)
        G = G_text
        pos = pos_text
        cluster_map = cluster_map_text
        st.info(f"Displaying a network of the top {st.session_state.max_nodes_to_display} most connected accounts.")
        st.info("Nodes are accounts, colored by content cluster. Edges show co-participation in a cluster.")

    elif coordination_mode == "Shared URLs":
        df_for_graph = df_for_analysis
        G_url, pos_url, cluster_map_url = cached_network_graph(df_for_graph, coordination_type="url", data_source=data_source + "_" + coordination_mode)
        G = G_url
        pos = pos_url
        cluster_map = cluster_map_url
        st.info(f"Displaying a network of the top {st.session_state.max_nodes_to_display} most connected accounts.")
        st.info("Nodes are accounts, colored by a grouping of shared URLs. Edges show co-sharing of URLs.")

    if not G.nodes():
        st.warning("No coordinated activity detected to build a network graph.")
    else:
        # Create Plotly figure
        fig_net = go.Figure()

        # Add edges as lines
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))

        # Add nodes with hover info
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            hover_text = f"User: {node}<br>Platform: {G.nodes[node].get('platform', 'N/A')}"
            node_text.append(hover_text)
            
            # --- FIX: Ensure numerical color for Plotly markers ---
            cluster_id = cluster_map.get(node)
            if isinstance(cluster_id, str):
                # Assign a unique numerical ID for string-based clusters (e.g., SharedURL_Group_X)
                # Use a simple hash or map to an integer for Plotly's colorscale
                node_color.append(hash(cluster_id) % 100) # Map string to a number for colorscale
            elif cluster_id not in [-1, -2]:
                node_color.append(cluster_id) # Use the numerical cluster ID directly
            else:
                node_color.append(-1) # Assign a distinct numerical value for 'No Coordination'

        # Create a DataFrame for node coloring and size
        nodes_df = pd.DataFrame({
            'x': node_x,
            'y': node_y,
            'text': node_text,
            'color': node_color,
            'size': [G.degree(node) for node in G.nodes()]
        })

        # Add nodes as markers
        fig_net.add_trace(go.Scatter(
            x=nodes_df['x'],
            y=nodes_df['y'],
            mode='markers',
            hoverinfo='text',
            text=nodes_df['text'],
            marker=dict(
                showscale=False, # Set to True if you want a color bar legend
                colorscale='Viridis', # Use a sequential or qualitative colorscale
                size=nodes_df['size'] * 3 + 5, # Scale size by degree
                color=nodes_df['color'], # Use the numerical color
                line_width=2,
                opacity=0.8
            ),
            name="Accounts"
        ))

        fig_net.update_layout(
            title='Network of Coordinated Accounts',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )

        st.plotly_chart(fig_net, use_container_width=True)

        st.markdown("### Risk & Influence Assessment")
        st.markdown("""
        **Centrality Analysis**: Accounts with high centrality (many connections) are key nodes in the network, potentially acting as amplifiers or originators of a message.
        - **Degree Centrality**: The number of connections a node has. High degree means an account is co-participating with many others.
        """)
        
        degree_centrality = nx.degree_centrality(G)
        risk_df = pd.DataFrame(degree_centrality.items(), columns=['Account', 'Degree Centrality'])
        risk_df = risk_df.sort_values(by='Degree Centrality', ascending=False).reset_index(drop=True)
        risk_df['Risk Score'] = (risk_df['Degree Centrality'] / risk_df['Degree Centrality'].max()) * 100
        risk_df = risk_df.merge(filtered_df_global[['account_id', 'Platform']].drop_duplicates(), left_on='Account', right_on='account_id', how='left').drop(columns='account_id')
        risk_df = risk_df.head(20)

        if not risk_df.empty:
            st.markdown("#### Top 20 Most Central Accounts (by Degree Centrality)")
            st.dataframe(risk_df, use_container_width=True)
            
            risk_csv = convert_df_to_csv(risk_df)
            st.download_button(
                "Download Risk Assessment CSV",
                risk_csv,
                "risk_assessment.csv",
                "text/csv",
                help="Downloads the list of accounts with their calculated risk scores."
            )
        else:
            st.warning("No network data available for risk assessment.")
