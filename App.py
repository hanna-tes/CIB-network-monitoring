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
from io import StringIO
import csv

# --- Set Page Config ---
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
    url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/TogoJULYData%20-%20Sheet1.csv"
    try:
        df = pd.read_csv(url)
        st.sidebar.success("✅ Default data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Failed to load default dataset: {e}")
        return pd.DataFrame()

# --- Preprocessing Function ---

def preprocess_data(df):
    """
    Preprocesses the DataFrame: maps columns, cleans text, parses timestamps.
    """

    # 1. Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # --- COLUMN MAPPING ---
    col_map = {
        # 💬 Text Content
        'Hit Sentence': 'text', 'Headline': 'text', 'message': 'text',
        'title': 'text', 'content': 'text', 'description': 'text',
        'opening text': 'text', 'Body': 'text', 'FullText': 'text',

        # 👤 Influencer / Author
        'Influencer': 'Influencer', 'author': 'Influencer', 'username': 'Influencer',
        'user': 'Influencer', 'authorMeta/name': 'Influencer', 'creator': 'Influencer',
        'authorname': 'Influencer',

        # 📅 Timestamps
        'Date': 'Timestamp', 'createTimeISO': 'Timestamp', 'published_date': 'Timestamp',
        'pubDate': 'Timestamp', 'created_at': 'Timestamp', 'Alternate Date Format': 'Timestamp',

        # 🔗 URL Variants
        'URL': 'URL', 'url': 'URL', 'webVideoUrl': 'URL', 'link': 'URL', 'post_url': 'URL',

        # 📺 Media & Channel Metadata
        'media_name': 'Outlet', 'channeltitle': 'Channel', 'source': 'Outlet',
        'Input Name': 'InputSource',
    }

    # Apply mapping
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

    # --- Validate Required Columns ---
    required_cols = ["Influencer", "Timestamp", "text"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        suggestions = {
            'Influencer': ['source', 'author', 'user', 'username'],
            'Timestamp': ['date', 'time', 'created', 'published'],
            'text': ['message', 'content', 'body', 'headline', 'hit sentence']
        }
        for col in missing_cols:
            close_matches = [c for c in df.columns if any(sugg in c.lower().replace(" ", "") for sugg in suggestions.get(col, []))]
            if close_matches:
                st.info(f"💡 Did you mean to rename {close_matches[0]} → {col}?")
            else:
                if col == "Influencer":
                    df['Influencer'] = "Unknown_User"
                elif col == "text":
                    st.error("🚫 No text column found. Cannot proceed.")
                    st.stop()
                elif col == "Timestamp":
                    df['Timestamp'] = pd.Timestamp.now(tz='UTC') # Ensure default timestamp is timezone aware
        # Re-check and stop if still missing critical columns after defaults
        for col in required_cols:
            if col not in df.columns:
                st.error(f"🛑 Still missing: '{col}' → Cannot continue.")
                st.stop()

    # --- Ensure 'text' column is string before stripping ---
    df['text'] = df['text'].astype(str)
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)

    # --- Timestamp Parsing ---
    date_formats = [
        '%b %d, %Y @ %H:%M:%S.%f',
        '%d-%b-%Y %I:%M%p',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f',
        '%d %b %Y %H:%M:%S',
        '%A, %d %b %Y %H:%M:%S',
        '%b %d, %Y %I:%M%p',
        '%d %b %Y %I:%M%p',
        '%Y-%m-%d %H:%M:%S%z',
        '%Y-%m-%d', # Date only
        '%m/%d/%Y', # Date only
        '%d %b %Y', # Date only
    ]

    def parse_timestamp_robust(timestamp):
        if pd.isna(timestamp):
            return pd.NaT
        if isinstance(timestamp, pd.Timestamp):
            return timestamp
        # Try direct conversion first for speed
        try:
            parsed = pd.to_datetime(timestamp, errors='coerce')
            if pd.notna(parsed):
                return parsed
        except (ValueError, TypeError):
            pass # Continue to specific formats

        # Then try specific formats
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce')
                if pd.notna(parsed):
                    return parsed
            except (ValueError, TypeError):
                continue
        return pd.NaT # Return NaT if no format matches

    df['Timestamp'] = df['Timestamp'].apply(parse_timestamp_robust)

    def localize_to_utc(dt):
        if pd.isna(dt):
            return dt
        # If timezone is naive, assume UTC and localize
        if dt.tzinfo is None:
            return dt.tz_localize('UTC')
        # If timezone aware, convert to UTC
        else:
            return dt.tz_convert('UTC')

    df['Timestamp'] = df['Timestamp'].apply(localize_to_utc)

    valid_ts = df['Timestamp'].notna().sum()
    st.info(f"✅ Parsed {valid_ts} valid timestamps.")

    # Drop rows with invalid timestamps AFTER all attempts to parse
    df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)

    # --- Create 'Platform' from URL ---
    url_cols = ['URL', 'url', 'webVideoUrl', 'link', 'post_url']
    url_found = False
    for col in url_cols:
        if col in df.columns:
            df['URL'] = df[col]
            url_found = True
            break

    if url_found and 'URL' in df.columns:
        df['Platform'] = df['URL'].apply(infer_platform_from_url)
    else:
        df['Platform'] = "Unknown"
        st.sidebar.warning("⚠️ No URL column found → all platforms marked as 'Unknown'")

    # --- Clean Text Further ---
    # Moved hashtag extraction before the more aggressive cleaning steps if needed
    # For now, keeping clean_text as is, and extracting hashtags directly from the 'text' column later in the app.
    # The current clean_text is fine, as it specifically targets 'rt @...' and URLs, not hashtags.
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'^QT.*?;.*', lambda m: m.group(0).split(';')[0], text)
        text = text.lower() # Converting to lower case *after* URL removal, to avoid issues.
        text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
        text = re.sub(r"\\n|\\r|\\t", " ", text) # Remove newline/tab characters
        text = re.sub(r"rt @\S+", "", text) # Remove RT mentions
        text = re.sub(r"qt @\S+", "", text) # Remove QT mentions
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text

    df['text'] = df['text'].apply(clean_text)

    # --- Extract original text (remove RT, specifically for similarity) ---
    # This is fine as it uses the *cleaned* 'text' column to get the original message.
    df['original_text'] = df['text'].apply(extract_original_text)

    # --- Final cleanup ---
    if df.empty:
        st.error("❌ No valid data after preprocessing.")
        st.stop()

    return df

# Vectorized similarity function
def find_textual_similarities(df, threshold=0.85):
    # Ensure 'original_text' is string type before dropping NaNs
    clean_df = df[['original_text', 'Influencer', 'Timestamp']].copy()
    clean_df['original_text'] = clean_df['original_text'].astype(str)
    clean_df = clean_df.dropna(subset=['original_text', 'Influencer', 'Timestamp'])
    clean_df = clean_df[clean_df['original_text'].str.strip() != ""]
    texts = clean_df['original_text'].tolist()

    if len(texts) < 2:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(texts)
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
        similar_pairs.append({
            'text1': row1['original_text'],
            'influencer1': row1['Influencer'],
            'time1': row1['Timestamp'],
            'text2': row2['original_text'],
            'influencer2': row2['Influencer'],
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
    try:
        from modules.clustering_utils import cluster_texts
        return cluster_texts(_df)
    except Exception as e:
        st.warning(f"Clustering module not found or failed to import. Falling back to dummy clustering. Error: {e}")
        _df = _df.copy()
        _df['cluster'] = 0 # Assign all to one cluster
        return _df

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df):
    try:
        from modules.clustering_utils import build_user_interaction_graph
        return build_user_interaction_graph(_df)
    except Exception as e:
        st.warning(f"Network graph module not found or failed to import. Falling back to dummy graph. Error: {e}")
        G = nx.Graph()
        nodes = _df['Influencer'].dropna().unique()
        if len(nodes) > 1:
            sampled_nodes = np.random.choice(nodes, min(len(nodes), 20), replace=False)
            for i in range(len(sampled_nodes)):
                G.add_node(sampled_nodes[i])
                if i > 0:
                    G.add_edge(sampled_nodes[i-1], sampled_nodes[i], weight=1)
            for _ in range(min(10, len(sampled_nodes) * (len(sampled_nodes) - 1) // 4)):
                u, v = np.random.choice(sampled_nodes, 2, replace=False)
                if not G.has_edge(u, v) and u !=v:
                    G.add_edge(u, v, weight=1)
        else:
            if len(nodes) == 1:
                G.add_node(nodes[0])

        pos = nx.spring_layout(G, seed=42)
        cluster_map = {n: 0 for n in G.nodes}
        return G, pos, cluster_map

# --- Load Data from URL ---
st.sidebar.header("📥 Data Source")
st.sidebar.info("Loading data from default URL...")
df = load_default_dataset()

# Exit if no data
if df is None or df.empty:
    st.warning("No data available. Please upload a CSV file or check the default URL.")
    st.stop()

# --- Preprocess ---
df = preprocess_data(df)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")

# 📅 Set default min/max dates for date filter
if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
    st.error("Timestamp column is not in datetime format after preprocessing. Cannot apply date filter.")
    st.stop()

min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

# 🧭 Date range filter
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
    start_dt = df['Timestamp'].min()
    end_dt = df['Timestamp'].max()


available_platforms = df['Platform'].dropna().astype(str).unique().tolist()
platforms = st.sidebar.multiselect(
    "Platforms",
    options=available_platforms,
    default=available_platforms
)

# Apply filters
filtered_df = df[
    (df['Timestamp'] >= start_dt) &
    (df['Timestamp'] <= end_dt) &
    (df['Platform'].isin(platforms))
].copy()

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust the date range or platforms.")
    st.stop()

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
    if not filtered_df.empty:
        top_influencers = filtered_df['Influencer'].value_counts().head(10)
        fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'Influencer'})
        st.plotly_chart(fig_src, use_container_width=True)

        # Plot for Platforms (instead of Outlets)
        # Ensure 'Platform' column exists and is used
        if 'Platform' in filtered_df.columns and not filtered_df['Platform'].empty:
            top_platforms = filtered_df['Platform'].value_counts().head(10)
            fig_platform = px.bar(top_platforms, title="Top 10 Platforms", labels={'value': 'Posts', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True)
        else:
            st.info("No 'Platform' column found or no data for platforms.")


        if 'Channel' in filtered_df.columns:
            top_channels = filtered_df['Channel'].value_counts().head(10)
            fig_chan = px.bar(top_channels, title="Top 10 Channels", labels={'value': 'Posts', 'index': 'Channel'})
            st.plotly_chart(fig_chan, use_container_width=True)
        else:
            st.info("No 'Channel' column found in data.")

        # Hashtag Extraction (moved creation of 'hashtags' column here, before use)
        # Ensure 'text' column is string before trying to find hashtags
        if 'text' in filtered_df.columns:
            filtered_df['hashtags'] = filtered_df['text'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
            all_hashtags = [tag for tags in filtered_df['hashtags'] for tag in tags if tags] # Filter out empty lists
            if all_hashtags:
                hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags", labels={'value': 'Frequency', 'index': 'Hashtag'})
                st.plotly_chart(fig_ht, use_container_width=True)
            else:
                st.info("No hashtags found in the filtered data.")
        else:
            st.info("No 'text' column found to extract hashtags.")


        time_series = filtered_df.set_index('Timestamp').resample('D').size()
        fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'Timestamp': 'Date'})
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No data available to display summary statistics.")

# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🧠 Narrative Detection & Coordination")
    MAX_ROWS = st.sidebar.slider("Max posts to analyze for similarity", 100, 1000, 300)
    # Ensure 'original_text' is ready for analysis, if not, create it before slicing
    if 'original_text' not in filtered_df.columns:
        filtered_df['original_text'] = filtered_df['text'].apply(extract_original_text)

    analysis_df = filtered_df.head(MAX_ROWS).copy()

    if analysis_df.empty:
        st.info("No data available for similarity analysis after applying filters and row limit.")
    else:
        with st.spinner(f"🔍 Finding coordinated narratives among {len(analysis_df)} posts..."):
            sim_df = cached_similarity_analysis(analysis_df, threshold=0.85)

        if not sim_df.empty:
            st.success(f"✅ Found {len(sim_df)} similar pairs.")
            narrative_summary = sim_df.groupby('shared_narrative').agg(
                share_count=('similarity', 'count'),
                influencers_involved=('influencer1', lambda x: ", ".join(x.astype(str).unique()[:5]) + ("..." if len(x.unique()) > 5 else ""))
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
        if 'original_text' not in filtered_df.columns:
            filtered_df['original_text'] = filtered_df['text'].apply(extract_original_text)

        clustered_df = cached_clustering(filtered_df)
        if 'cluster' not in clustered_df.columns:
            st.warning("⚠️ Clustering did not return 'cluster' column. Displaying unclustered data.")
            clustered_df['cluster'] = "N/A"

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
            st.info("No clusters detected or no data available for clustering.")

    except Exception as e:
        st.warning(f"⚠️ Clustering analysis failed: {e}")

    st.markdown("### 🕸️ User Interaction Network")
    try:
        G, pos, cluster_map = cached_network_graph(clustered_df if 'clustered_df' in locals() and not clustered_df.empty else filtered_df)

        if not G.nodes():
            st.info("No nodes to display in the network graph. This might be due to filtered data or issues in graph creation.")
        else:
            edge_trace = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'))

            node_colors = [cluster_map.get(node, 0) for node in G.nodes()]
            if len(set(node_colors)) > 1:
                marker_colorscale = 'Set3'
            else:
                marker_colorscale = 'Blues'

            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                text=[f"Influencer: {node}<br>Cluster: {cluster_map.get(node, 'N/A')}" for node in G.nodes()],
                mode='markers+text',
                textposition="top center",
                marker=dict(
                    size=12,
                    color=node_colors,
                    colorscale=marker_colorscale,
                    colorbar=dict(title="Clusters") if len(set(node_colors)) > 1 else None,
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

    st.markdown("### ⚠️ High-Risk Influencers")
    try:
        if 'sim_df' in locals() and not sim_df.empty:
            all_influencers = pd.concat([
                sim_df[['influencer1']].rename(columns={'influencer1': 'Influencer'}),
                sim_df[['influencer2']].rename(columns={'influencer2': 'Influencer'})
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
