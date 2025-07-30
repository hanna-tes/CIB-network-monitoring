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
def combine_social_media_data(meltwater_df, civicsignals_df, openmeasure_df=None):
    combined_dfs = []

    def get_specific_col(df, col_name_lower):
        if col_name_lower in df.columns:
            return df[col_name_lower]
        return pd.Series([np.nan] * len(df), index=df.index)

    if meltwater_df is not None and not meltwater_df.empty:
        meltwater_df.columns = meltwater_df.columns.str.lower()
        mw = pd.DataFrame()
        mw['account_id'] = get_specific_col(meltwater_df, 'influencer')
        mw['content_id'] = get_specific_col(meltwater_df, 'tweet id')
        mw['object_id'] = get_specific_col(meltwater_df, 'hit sentence')
        mw['original_url'] = get_specific_col(meltwater_df, 'url')
        mw['timestamp_share'] = get_specific_col(meltwater_df, 'date')
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)

    if civicsignals_df is not None and not civicsignals_df.empty:
        civicsignals_df.columns = civicsignals_df.columns.str.lower()
        cs = pd.DataFrame()
        cs['account_id'] = get_specific_col(civicsignals_df, 'media_name')
        cs['content_id'] = get_specific_col(civicsignals_df, 'stories_id')
        cs['object_id'] = get_specific_col(civicsignals_df, 'title')
        cs['original_url'] = get_specific_col(civicsignals_df, 'url')
        cs['timestamp_share'] = get_specific_col(civicsignals_df, 'publish_date')
        cs['source_dataset'] = 'CivicSignals'
        combined_dfs.append(cs)

    if openmeasure_df is not None and not openmeasure_df.empty:
        openmeasure_df.columns = openmeasure_df.columns.str.lower()
        om = pd.DataFrame()
        om['account_id'] = get_specific_col(openmeasure_df, 'actor_username')
        om['content_id'] = get_specific_col(openmeasure_df, 'id')
        om['object_id'] = get_specific_col(openmeasure_df, 'text')
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

    date_formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p',
        '%A, %d %b %Y %H:%M:%S', '%b %d, %I:%M%p', '%d %b %Y %I:%M%p',
        '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y',
    ]

    def parse_timestamp_robust(timestamp):
        if pd.isna(timestamp): return pd.NaT
        if isinstance(timestamp, (int, float)): return pd.NaT
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
    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')
    combined = combined[combined['object_id'].str.strip() != ""].copy()
    combined = combined.drop_duplicates(subset=['account_id', 'content_id', 'object_id', 'timestamp_share']).reset_index(drop=True)
    return combined

# --- Final Preprocessing Function ---
def final_preprocess_and_map_columns(df):
    if df.empty:
        return df
    df_processed = df.copy()
    df_processed.rename(columns={'original_url': 'URL'}, inplace=True)
    df_processed['object_id'] = df_processed['object_id'].astype(str)
    df_processed = df_processed[df_processed['object_id'].notna()]
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""]
    df_processed = df_processed[df_processed['object_id'].str.lower() != "nan"].reset_index(drop=True)

    def clean_text_for_display(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    df_processed['object_id'] = df_processed['object_id'].apply(clean_text_for_display)
    df_processed = df_processed[df_processed['object_id'].str.len() > 0].reset_index(drop=True)

    initial_rows_before_filter = len(df_processed)
    df_processed = df_processed[
        ~df_processed['object_id'].str.startswith('rt @') &
        ~df_processed['object_id'].str.startswith('qt @')
    ].reset_index(drop=True)
    filtered_count = initial_rows_before_filter - len(df_processed)
    if filtered_count > 0:
        st.info(f"Filtered out {filtered_count} retweets/quoted tweets.")

    df_processed['timestamp_share'] = pd.to_datetime(df_processed['timestamp_share'], errors='coerce', utc=True)
    df_processed = df_processed.dropna(subset=['timestamp_share']).reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)

    if 'Outlet' not in df_processed.columns:
        df_processed['Outlet'] = np.nan
    if 'Channel' not in df_processed.columns:
        df_processed['Channel'] = np.nan

    if df_processed.empty:
        st.error("❌ No valid data after preprocessing.")
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

# --- Cached Functions with data_source to prevent cache reuse ---
@st.cache_data(show_spinner="🔍 Computing textual similarities...")
def cached_similarity_analysis(_df, threshold=0.85, data_source="default"):
    return find_textual_similarities(_df, threshold)

@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df, data_source="default"):
    return cluster_texts(_df)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df_for_graph, coordination_type="text", data_source="default"):
    return build_user_interaction_graph(_df_for_graph, coordination_type)

# --- Sidebar: Data Source Selection ---
st.sidebar.header("📥 Data Source")
data_source = st.sidebar.radio("Choose data source:", ("Use Default Datasets", "Upload CSV Files"))

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
            meltwater_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',', low_memory=False)
            st.sidebar.success("✅ Meltwater CSV loaded.")
        except Exception as e:
            st.error(f"❌ Failed to read Meltwater CSV: {e}")
            st.stop()

    if uploaded_civicsignals and uploaded_civicsignals.type != "application/zip":
        bytes_data = uploaded_civicsignals.getvalue()
        try:
            civicsignals_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',')
            st.sidebar.success("✅ CivicSignals CSV loaded.")
        except Exception as e:
            st.error(f"❌ Failed to read CivicSignals CSV: {e}")
            st.stop()

    if uploaded_openmeasure:
        bytes_data = uploaded_openmeasure.getvalue()
        try:
            openmeasure_df_upload = pd.read_csv(BytesIO(bytes_data), sep=',', low_memory=False)
            st.sidebar.success("✅ Open-Measure CSV loaded.")
        except Exception as e:
            st.error(f"❌ Failed to read Open-Measure CSV: {e}")
            st.stop()

    if not meltwater_df_upload.empty or not civicsignals_df_upload.empty or not openmeasure_df_upload.empty:
        with st.spinner("Combining uploaded datasets..."):
            combined_raw_df = combine_social_media_data(
                meltwater_df_upload if not meltwater_df_upload.empty else None,
                civicsignals_df_upload if not civicsignals_df_upload.empty else None,
                openmeasure_df_upload if not openmeasure_df_upload.empty else None
            )
        st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from uploaded files.")
    else:
        st.warning("Please upload at least one CSV file to proceed.")
        st.stop()

# --- Final Preprocess ---
with st.spinner("⏳ Preprocessing and mapping combined data..."):
    df = final_preprocess_and_map_columns(combined_raw_df)

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
        "preprocessed_combined_core_data.csv",
        "text/csv",
        help="Downloads the data after all preprocessing and column mapping."
    )
else:
    st.sidebar.warning("Could not create downloadable dataset.")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters (Apply to all tabs)")
if not pd.api.types.is_datetime64_any_dtype(df['timestamp_share']):
    st.error("timestamp_share column is not in datetime format.")
    st.stop()

min_date = df['timestamp_share'].min().date()
max_date = df['timestamp_share'].max().date()
selected_date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

if len(selected_date_range) == 2:
    start_dt = pd.Timestamp(selected_date_range[0], tz='UTC')
    end_dt = pd.Timestamp(selected_date_range[1], tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
else:
    start_dt = pd.Timestamp(selected_date_range[0], tz='UTC')
    end_dt = start_dt + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

available_platforms_global = df['Platform'].dropna().astype(str).unique().tolist()
platforms_global = st.sidebar.multiselect("Platforms", options=available_platforms_global, default=available_platforms_global)

filtered_df_global = df[
    (df['timestamp_share'] >= start_dt) &
    (df['timestamp_share'] <= end_dt) &
    (df['Platform'].isin(platforms_global))
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
    display_cols_overview = ['account_id', 'content_id', 'object_id', 'timestamp_share']
    existing_cols = [col for col in df.columns if col in display_cols_overview]
    st.dataframe(df[existing_cols].head(10))

    st.markdown("---")
    if 'source_dataset' in filtered_df_global.columns:
        st.markdown("### 📊 Data Sources in Filtered Data")
        source_counts = filtered_df_global['source_dataset'].value_counts()
        st.dataframe(source_counts)

    if not filtered_df_global.empty:
        top_influencers = filtered_df_global['account_id'].value_counts().head(10)
        fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'account_id'})
        st.plotly_chart(fig_src, use_container_width=True)

        if 'Platform' in filtered_df_global.columns:
            all_platforms_counts = filtered_df_global['Platform'].value_counts()
            fig_platform = px.bar(all_platforms_counts, title="Post Distribution by Platform", labels={'value': 'Posts', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True)

        if 'Outlet' in filtered_df_global.columns and filtered_df_global['Outlet'].notna().any():
            top_outlets = filtered_df_global['Outlet'].value_counts().head(10)
            fig_outlet = px.bar(top_outlets, title="Top 10 Media Outlets/Channels", labels={'value': 'Posts', 'index': 'Outlet'})
            st.plotly_chart(fig_outlet, use_container_width=True)
        elif 'Channel' in filtered_df_global.columns and filtered_df_global['Channel'].notna().any():
            top_channels = filtered_df_global['Channel'].value_counts().head(10)
            fig_chan = px.bar(top_channels, title="Top 10 Channels", labels={'value': 'Posts', 'index': 'Channel'})
            st.plotly_chart(fig_chan, use_container_width=True)

        social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media'])].copy()
        if not social_media_df.empty and 'object_id' in social_media_df.columns:
            social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
            all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
            if all_hashtags:
                hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                st.plotly_chart(fig_ht, use_container_width=True)

        time_series = filtered_df_global.set_index('timestamp_share').resample('D').size()
        fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'timestamp_share': 'Date'})
        st.plotly_chart(fig_ts, use_container_width=True)

# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🧠 Narrative Detection & Coordination")
    st.markdown("This section identifies coordinated narratives using text similarity.")
    st.markdown("---")

    platforms_analysis = st.multiselect(
        "Platforms to include in Similarity Analysis:",
        options=filtered_df_global['Platform'].dropna().astype(str).unique().tolist(),
        default=filtered_df_global['Platform'].dropna().astype(str).unique().tolist(),
        key="platforms_analysis_tab2"
    )

    analysis_df_filtered_by_platform = filtered_df_global[filtered_df_global['Platform'].isin(platforms_analysis)].copy()
    MAX_ROWS_SIMILARITY = st.slider("Max posts to analyze for similarity", 100, 1000, 300, key="max_rows_similarity")

    analysis_df = analysis_df_filtered_by_platform[
        (analysis_df_filtered_by_platform['original_text'].notna()) &
        (analysis_df_filtered_by_platform['original_text'].astype(str).str.strip() != "")
    ].head(MAX_ROWS_SIMILARITY).copy()

    sim_df = pd.DataFrame()
    if not analysis_df.empty:
        with st.spinner(f"🔍 Finding coordinated narratives among {len(analysis_df)} original posts..."):
            sim_df = cached_similarity_analysis(
                analysis_df,
                threshold=0.85,
                data_source="upload" if data_source == "Upload CSV Files" else "default"
            )
        if not sim_df.empty:
            st.success(f"✅ Found {len(sim_df)} similar pairs.")
            narrative_summary = sim_df.groupby('shared_narrative').agg(
                share_count=('similarity', 'count'),
                account_ids_involved=('account_id1', lambda x: ", ".join(x.astype(str).unique()[:5]) + ("..." if len(x.unique()) > 5 else ""))
            ).sort_values(by='share_count', ascending=False).reset_index()

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
            st.markdown("### Full Similarity Pairs")
            display_sim_df = sim_df[['text1', 'account_id1', 'platform1', 'timestamp_share1', 'url1', 'text2', 'account_id2', 'platform2', 'timestamp_share2', 'url2', 'similarity']].copy()
            display_sim_df['url1'] = display_sim_df['url1'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) and x.startswith('http') else '')
            display_sim_df['url2'] = display_sim_df['url2'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) and x.startswith('http') else '')
            st.markdown(display_sim_df.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.info("No significant similarities found.")
    else:
        st.info("No valid text data for similarity analysis.")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")
    st.markdown("---")

    platforms_network = st.multiselect(
        "Platforms to include in Network & Risk Analysis:",
        options=filtered_df_global['Platform'].dropna().astype(str).unique().tolist(),
        default=filtered_df_global['Platform'].dropna().astype(str).unique().tolist(),
        key="platforms_network_tab3"
    )

    network_df_filtered_by_platform = filtered_df_global[filtered_df_global['Platform'].isin(platforms_network)].copy()
    coordination_basis = st.radio(
        "Choose basis for Coordination and Network Analysis:",
        ("Text Content (Narrative Similarity)", "Shared URLs"),
        key="coordination_basis_selector"
    )

    clustered_df = pd.DataFrame()
    G, pos, cluster_map = nx.Graph(), {}, {}

    if coordination_basis == "Text Content (Narrative Similarity)":
        df_for_clustering = network_df_filtered_by_platform[
            (network_df_filtered_by_platform['original_text'].notna()) &
            (network_df_filtered_by_platform['original_text'].astype(str).str.strip() != "")
        ].copy()
        if not df_for_clustering.empty:
            clustered_df = cached_clustering(
                df_for_clustering,
                data_source="upload" if data_source == "Upload CSV Files" else "default"
            )
        else:
            st.info("No valid text data for clustering.")
    # URL-based handled later

    max_influencers_graph = st.slider("Max Influencers for Network Graph", 10, 200, 50, key="max_influencers_graph")
    graph_df_subset = None

    if coordination_basis == "Text Content (Narrative Similarity)" and not clustered_df.empty:
        top_active_influencers = clustered_df['account_id'].value_counts().nlargest(max_influencers_graph).index.tolist()
        graph_df_subset = clustered_df[clustered_df['account_id'].isin(top_active_influencers)].copy()
    elif coordination_basis == "Shared URLs":
        graph_df_base = network_df_filtered_by_platform[
            (network_df_filtered_by_platform['URL'].notna()) &
            (network_df_filtered_by_platform['URL'].str.strip() != '')
        ].copy()
        top_active_influencers = graph_df_base['account_id'].value_counts().nlargest(max_influencers_graph).index.tolist()
        graph_df_subset = graph_df_base[graph_df_base['account_id'].isin(top_active_influencers)].copy()

    if graph_df_subset is not None and not graph_df_subset.empty:
        G, pos, cluster_map = cached_network_graph(
            graph_df_subset,
            coordination_type="text" if coordination_basis == "Text Content (Narrative Similarity)" else "url",
            data_source="upload" if data_source == "Upload CSV Files" else "default"
        )
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
            text=[f"Influencer: {node}<br>Posts: {influencer_post_counts.get(node, 0)}<br>Group: {G.nodes[node].get('cluster', 'N/A')}<br>Platform: {G.nodes[node].get('platform', 'N/A')}" for node in G.nodes()],
            mode='markers+text',
            textposition="top center",
            marker=dict(size=node_sizes_raw, color=node_color_vals, line=dict(width=2, color='darkblue')),
            hoverinfo='text'
        )

        fig_net = go.Figure(data=edge_trace + [node_trace],
                            layout=go.Layout(
                                title="Influencer Coordination Network",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=60),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=600))
        st.plotly_chart(fig_net, use_container_width=True)

        if unique_clusters:
            st.markdown("#### Group Color Legend:")
            legend_items = [f"<span style='color:{color_map_for_plot[cluster_id]}'>●</span> Group: {cluster_id}" for cluster_id in unique_clusters]
            st.markdown("<br>".join(legend_items), unsafe_allow_html=True)
    else:
        st.info("No data available to build the network graph.")

    st.markdown("### ⚠️ High-Risk Influencers")
    if 'sim_df' in locals() and not sim_df.empty:
        all_influencers = pd.concat([
            sim_df[['account_id1']].rename(columns={'account_id1': 'account_id'}),
            sim_df[['account_id2']].rename(columns={'account_id2': 'account_id'})
        ])['account_id'].dropna().value_counts()
        high_risk = all_influencers[all_influencers >= 3]
        if not high_risk.empty:
            fig_hr = px.bar(high_risk, title="Influencers in ≥3 Coordinated Messages", labels={'value': 'Coordination Instances', 'index': 'account_id'}, color='value', color_continuous_scale='Reds')
            st.plotly_chart(fig_hr, use_container_width=True)
        else:
            st.info("No influencers in 3+ coordinated messages.")
    else:
        st.info("No coordinated narratives detected.")
