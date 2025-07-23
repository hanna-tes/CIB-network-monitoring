import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx 
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN # Import DBSCAN
from itertools import combinations # Import combinations
import re
from io import StringIO
import csv

# --- Set Page Config ---
st.set_page_config(page_title="CIB Dashboard", layout="wide")
st.title("🕵️ CIB Network Monitoring Dashboard")

# --- Helper Functions ---
def infer_platform_from_url(url):
    """Infers the social media or media platform from a given URL."""
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    if "tiktok.com" in url:
        return "TikTok"
    elif "facebook.com" in url or "fb.watch" in url:
        return "Facebook"
    elif "twitter.com" in url or "x.com" in url:
        return "X" # Changed to X as per user's typical usage
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
    """Removes 'RT @user:' prefix to get the core message for similarity analysis."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^RT\s+@\w+:\s*', '', text).strip()
    return cleaned

@st.cache_data(show_spinner=False)
def load_default_dataset():
    """Loads the default dataset from a specified CSV file or URL."""
    file_name = "TogoJULYData - Sheet1.csv"
    try:
        df = pd.read_csv(file_name)
        st.sidebar.success(f"✅ Default data loaded successfully from {file_name}.")
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_name}. Please ensure the default data file is in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load default dataset from {file_name}: {e}")
        return pd.DataFrame()

# --- Preprocessing Function ---
def preprocess_data(df):
    """
    Preprocesses the DataFrame: maps columns, creates 'text' column, cleans text,
    parses and localizes timestamps, and infers platform.
    """
    # 1. Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # --- COLUMN MAPPING ---
    col_map = {
        'Hit Sentence': 'Hit Sentence',
        'Headline': 'Headline',
        'message': 'message',
        'title': 'title',
        'content': 'content',
        'description': 'description',
        'opening text': 'Opening Text',
        'Body': 'Body',
        'FullText': 'FullText',

        'Influencer': 'Influencer_Candidate_1',
        'author': 'Influencer_Candidate_2',
        'username': 'Influencer_Candidate_3',
        'user': 'Influencer_Candidate_4',
        'authorMeta/name': 'Influencer_Candidate_5',
        'creator': 'Influencer_Candidate_6',
        'authorname': 'Influencer_Candidate_7',
        'Source': 'Influencer_Candidate_8', # User's 'Source' column

        'Date': 'Timestamp', 'createTimeISO': 'Timestamp', 'published_date': 'Timestamp',
        'pubDate': 'Timestamp', 'created_at': 'Timestamp', 'Alternate Date Format': 'Timestamp',

        'URL': 'URL', 'url': 'URL', 'webVideoUrl': 'URL', 'link': 'URL', 'post_url': 'URL',

        'media_name': 'Outlet', # Map media_name to Outlet
        'channeltitle': 'Channel',
        'source': 'Outlet', # Another common name for media source
        'Input Name': 'InputSource',
    }

    original_cols = df.columns.tolist()
    new_columns_dict = {}
    for col in original_cols:
        matched = False
        if col in col_map:
            new_columns_dict[col] = col_map[col]
            matched = True
        else:
            normalized_col = col.lower().replace(" ", "").replace("_", "").replace("-", "")
            for key, target in col_map.items():
                norm_key = key.lower().replace(" ", "").replace("_", "").replace("-", "")
                if normalized_col == norm_key:
                    new_columns_dict[col] = target
                    matched = True
                    break
        if not matched and col not in new_columns_dict:
            new_columns_dict[col] = col

    df = df.rename(columns=new_columns_dict)
    df = df.loc[:,~df.columns.duplicated()] # Remove truly duplicate column names after mapping

    # --- Create 'text' column, prioritizing 'Hit Sentence' ---
    df['text'] = ''
    found_primary_text_col = False

    if 'Hit Sentence' in df.columns:
        df['text'] = df['Hit Sentence'].astype(str).replace('nan', np.nan).fillna('')
        found_primary_text_col = True
    else:
        st.warning("⚠️ 'Hit Sentence' column not found. Falling back to other text candidates.")

    if not found_primary_text_col or df['text'].astype(str).str.strip().eq('').all():
        text_candidates_fallback = ['Opening Text', 'Headline', 'message', 'title', 'content', 'description', 'Body', 'FullText']
        for col in text_candidates_fallback:
            if col in df.columns and not df[col].empty:
                df['text'] = df[col].astype(str).replace('nan', np.nan).fillna('')
                found_primary_text_col = True
                break

    if not found_primary_text_col:
        st.warning("⚠️ No suitable text column found among candidates. 'text' column might be empty.")
        df['text'] = ""

    df['text'] = df['text'].astype(str).replace('nan', np.nan)
    df = df.dropna(subset=['text']).reset_index(drop=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)

    # --- Populate 'Influencer' column with best available data ---
    df['Influencer'] = "Unknown_User" # Default to 'Unknown_User'

    influencer_candidates = [
        'Influencer_Candidate_1', 'Influencer_Candidate_2', 'Influencer_Candidate_3',
        'Influencer_Candidate_4', 'Influencer_Candidate_5', 'Influencer_Candidate_6',
        'Influencer_Candidate_7', 'Influencer_Candidate_8'
    ]

    for cand_col in influencer_candidates:
        if cand_col in df.columns and not df[cand_col].astype(str).str.strip().eq('').all():
            df['Influencer'] = df['Influencer'].mask(
                (df['Influencer'] == "Unknown_User") | df['Influencer'].astype(str).str.strip().eq(''),
                df[cand_col].astype(str).replace('nan', np.nan).fillna('Unknown_User')
            )

    # Final fallback for any remaining 'Unknown_User' or truly empty influencer fields with 'Outlet'
    if 'Outlet' in df.columns:
        df['Influencer'] = df['Influencer'].mask(
            (df['Influencer'] == "Unknown_User") | df['Influencer'].astype(str).str.strip().eq(''),
            df['Outlet'].astype(str).replace('nan', np.nan).fillna('Unknown_User')
        )

    # Drop temporary influencer candidate columns
    cols_to_drop = [col for col in influencer_candidates if col in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Ensure Influencer column is always present and clean (already handled above but final check)
    df['Influencer'] = df['Influencer'].astype(str).replace('nan', np.nan).fillna('Unknown_User')


    # --- Timestamp Parsing ---
    date_formats = [
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f', '%d %b %Y %H:%M:%S', '%A, %d %b %Y %H:%M:%S',
        '%b %d, %Y %I:%M%p', '%d %b %Y %I:%M%p', '%Y-%m-%d %H:%M:%S%z',
        '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y',
    ]

    def parse_timestamp_robust(timestamp):
        if pd.isna(timestamp):
            return pd.NaT
        if isinstance(timestamp, pd.Timestamp):
            return timestamp
        try:
            parsed = pd.to_datetime(timestamp, errors='coerce')
            if pd.notna(parsed): return parsed
        except (ValueError, TypeError): pass

        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce')
                if pd.notna(parsed): return parsed
            except (ValueError, TypeError): continue
        return pd.NaT

    df['Timestamp'] = df['Timestamp'].apply(parse_timestamp_robust)

    def localize_to_utc(dt):
        if pd.isna(dt): return dt
        if dt.tzinfo is None: return dt.tz_localize('UTC')
        else: return dt.tz_convert('UTC')

    df['Timestamp'] = df['Timestamp'].apply(localize_to_utc)
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

    # --- Clean Text Further (after 'text' column is finalized) ---
    def clean_text_final(text):
        """Applies final cleaning to the 'text' column, preserving hashtags."""
        if not isinstance(text, str): return ""
        text = re.sub(r'^QT.*?;.*', lambda m: m.group(0).split(';')[0], text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r"rt @\S+", "", text)
        text = re.sub(r"qt @\S+", "", text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['text'] = df['text'].apply(clean_text_final)

    # --- Extract original text (for similarity, removes RT specifically) ---
    df['original_text'] = df['text'].apply(extract_original_text)

    # --- Final check for empty DataFrame ---
    if df.empty:
        st.error("❌ No valid data after complete preprocessing.")
        st.stop()

    return df

# Vectorized similarity function
def find_textual_similarities(df, threshold=0.85):
    """
    Computes cosine similarity between 'original_text' entries to find similar pairs,
    including URLs for context.
    """
    clean_df = df[['original_text', 'Influencer', 'Timestamp', 'URL']].copy()
    clean_df['original_text'] = clean_df['original_text'].astype(str)
    clean_df = clean_df.dropna(subset=['original_text', 'Influencer', 'Timestamp'])
    clean_df = clean_df[clean_df['original_text'].str.strip() != ""]
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
        narrative_snippet = row1['original_text'][:150]
        if len(row1['original_text']) > 150:
            narrative_snippet += "..."
        if not narrative_snippet.strip():
            narrative_snippet = "Empty/Cleaned Text"

        similar_pairs.append({
            'text1': row1['original_text'],
            'influencer1': row1['Influencer'],
            'time1': row1['Timestamp'],
            'url1': row1['URL'],
            'text2': row2['original_text'],
            'influencer2': row2['Influencer'],
            'time2': row2['Timestamp'],
            'url2': row2['URL'],
            'similarity': round(sim_matrix[i, j], 3),
            'shared_narrative': narrative_snippet
        })
    return pd.DataFrame(similar_pairs)

# --- Clustering and Graph Building Functions (from user's modules.clustering_utils) ---
def cluster_texts(df, eps=0.3, min_samples=2):
    # Ensure 'original_text' exists before vectorizing
    if 'original_text' not in df.columns:
        df['original_text'] = df['text'].apply(extract_original_text) # Fallback in case it's not present

    texts_to_cluster = df['original_text'].astype(str).tolist()
    
    if not texts_to_cluster or all(text.strip() == "" for text in texts_to_cluster):
        st.warning("No valid text data for clustering. Assigning all to cluster 0.")
        df_copy = df.copy()
        df_copy['cluster'] = 0
        return df_copy

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts_to_cluster)
    except ValueError as e:
        st.warning(f"Could not create TF-IDF matrix for clustering: {e}. Assigning all to cluster 0.")
        df_copy = df.copy()
        df_copy['cluster'] = 0
        return df_copy
        
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(tfidf_matrix)
    df_copy = df.copy()
    df_copy['cluster'] = clustering.labels_
    return df_copy

def build_user_interaction_graph(df):
    G = nx.Graph()
    # Simple: connect users who share same narrative (via high similarity or same cluster)
    # Ensure 'Influencer' is used as the node name, not 'Source' as in the original module
    grouped = df.groupby('cluster')
    for cluster_id, group in grouped:
        if cluster_id == -1 or len(group) < 2: # -1 is noise, or too few members
            continue
        users = group['Influencer'].dropna().unique().tolist() # Use 'Influencer'
        for u1, u2 in combinations(users, 2):
            if G.has_edge(u1, u2):
                G[u1][u2]['weight'] += 1
            else:
                G.add_edge(u1, u2, weight=1)
    
    # Create position for all nodes, including isolated ones if needed
    all_influencers = df['Influencer'].dropna().unique().tolist()
    for inf in all_influencers:
        if inf not in G.nodes():
            G.add_node(inf)

    # Use a layout that handles disconnected components well
    pos = nx.spring_layout(G, seed=42, k=0.1, iterations=50) # Adjust k and iterations for better spread

    # Create a cluster map for coloring nodes
    # If a user is in multiple clusters (e.g. if the original_text has multiple posts), pick the first one
    cluster_map = df.set_index('Influencer')['cluster'].to_dict()
    # Handle cases where an influencer might be in multiple clusters in the raw data,
    # or not present in 'cluster' column if filtered out by text processing.
    # Ensure every node in G has a cluster ID. Default to 0 if not found.
    final_cluster_map = {node: cluster_map.get(node, 0) for node in G.nodes()}

    return G, pos, final_cluster_map


# --- Cached Expensive Functions ---
@st.cache_data(show_spinner="🔍 Computing textual similarities...")
def cached_similarity_analysis(_df, threshold=0.85):
    return find_textual_similarities(_df, threshold)

@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df):
    """
    Performs text clustering using the integrated DBSCAN clustering function.
    """
    # Directly call the defined cluster_texts function
    return cluster_texts(_df)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df):
    """
    Builds a user interaction network graph using the integrated function.
    """
    # Directly call the defined build_user_interaction_graph function
    return build_user_interaction_graph(_df)

# --- Data Source Selection ---
st.sidebar.header("📥 Data Source")
data_source_option = st.sidebar.radio(
    "Choose data source:",
    ("Use Default Data", "Upload CSV")
)

df = pd.DataFrame() # Initialize df to an empty DataFrame

if data_source_option == "Use Default Data":
    df = load_default_dataset()
elif data_source_option == "Upload CSV":
    # Allow multiple file uploads
    uploaded_files = st.sidebar.file_uploader("Upload your CSV file(s)", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        dfs_from_upload = []
        for uploaded_file in uploaded_files:
            try:
                df_temp = pd.read_csv(uploaded_file)
                dfs_from_upload.append(df_temp)
                st.sidebar.success(f"✅ CSV '{uploaded_file.name}' uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading CSV file '{uploaded_file.name}': {e}")
        if dfs_from_upload:
            df = pd.concat(dfs_from_upload, ignore_index=True)
            st.sidebar.info(f"Combined data from {len(dfs_from_upload)} file(s).")
        else:
            st.error("No valid CSV files were uploaded or could be processed.")
            df = pd.DataFrame() # Ensure df is empty if an error occurs
    else:
        df = pd.DataFrame() # Ensure df is empty if no files are uploaded

# Exit if no data after selection
if df is None or df.empty:
    st.warning("No data available. Please select a data source and ensure it's valid.")
    st.stop()

# --- Preprocess ---
with st.spinner("⏳ Preprocessing data..."):
    df = preprocess_data(df)

# Exit if no data after preprocessing
if df.empty:
    st.warning("No valid data available after preprocessing. Please check your data file and column mappings.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")

if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
    st.error("Timestamp column is not in datetime format after preprocessing. Cannot apply date filter.")
    st.stop()

min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

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
        st.write("This chart shows the top 10 influencers by the number of posts in the filtered dataset.")
        top_influencers = filtered_df['Influencer'].value_counts().head(10)
        fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'Influencer'})
        st.plotly_chart(fig_src, use_container_width=True)

        if 'Platform' in filtered_df.columns and not filtered_df['Platform'].empty:
            st.write("This chart displays the distribution of posts across all identified social media and media platforms in the dataset.")
            # Changed to show all platforms, not just top 10
            all_platforms_counts = filtered_df['Platform'].value_counts()
            fig_platform = px.bar(all_platforms_counts, title="Post Distribution by Platform", labels={'value': 'Posts', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True)
        else:
            st.info("No 'Platform' column found or no data for platforms. This typically happens if no URLs are present in the data.")

        if 'Channel' in filtered_df.columns:
            st.write("This chart illustrates the top 10 channels where content was published.")
            top_channels = filtered_df['Channel'].value_counts().head(10)
            fig_chan = px.bar(top_channels, title="Top 10 Channels", labels={'value': 'Posts', 'index': 'Channel'})
            st.plotly_chart(fig_chan, use_container_width=True)

        if 'text' in filtered_df.columns and not filtered_df['text'].empty:
            st.write("This chart highlights the top 10 most frequently used hashtags in the filtered posts.")
            filtered_df['hashtags'] = filtered_df['text'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])

            all_hashtags = [tag for tags_list in filtered_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list if tags_list]

            if all_hashtags:
                hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags", labels={'value': 'Frequency', 'index': 'Hashtag'})
                st.plotly_chart(fig_ht, use_container_width=True)
            else:
                st.info("No hashtags found in the filtered data 'text' column.")
        else:
            st.info("No 'text' column found or it's empty to extract hashtags.")

        st.write("This area chart visualizes the daily volume of posts over the selected date range.")
        time_series = filtered_df.set_index('Timestamp').resample('D').size()
        fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'Timestamp': 'Date'})
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No data available to display summary statistics.")

# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🧠 Narrative Detection & Coordination")
    MAX_ROWS = st.sidebar.slider("Max posts to analyze for similarity", 100, 1000, 300)
    if 'original_text' not in filtered_df.columns:
        filtered_df['original_text'] = filtered_df['text'].apply(extract_original_text)

    analysis_df = filtered_df[filtered_df['original_text'].astype(str).str.strip() != ""].head(MAX_ROWS).copy()

    if analysis_df.empty:
        st.info("No valid text data available for similarity analysis after applying filters and row limit.")
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

            st.write("This table summarizes the top coordinated narratives, including the number of shares and involved influencers.")
            st.dataframe(narrative_summary)
            st.markdown("### 🔄 Full Similarity Pairs")
            st.write("This table lists all detected pairs of similar texts, along with their influencers, timestamps, similarity scores, and links to the original posts for verification.")
            
            # Create a display DataFrame with formatted URLs
            display_sim_df = sim_df.drop(columns=['shared_narrative'], errors='ignore').copy()
            # Convert URLs to clickable links
            display_sim_df['url1'] = display_sim_df['url1'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) else '')
            display_sim_df['url2'] = display_sim_df['url2'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) else '')
            
            # Render as HTML to make links clickable
            st.markdown(display_sim_df.to_html(escape=False), unsafe_allow_html=True)

        else:
            st.info("No significant similarities found above threshold.")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")
    try:
        if 'original_text' not in filtered_df.columns:
            filtered_df['original_text'] = filtered_df['text'].apply(extract_original_text)

        df_for_clustering = filtered_df[filtered_df['text'].astype(str).str.strip() != ""].copy()
        if df_for_clustering.empty:
            st.info("No valid text data for clustering analysis.")
            clustered_df = pd.DataFrame()
        else:
            clustered_df = cached_clustering(df_for_clustering)
            # No need for the warning if 'cluster' column is missing, as the functions themselves handle fallbacks
            # if 'cluster' not in clustered_df.columns:
            #     st.warning("⚠️ Clustering did not return 'cluster' column. Displaying unclustered data.")
            #     clustered_df['cluster'] = "N/A"

        if not clustered_df.empty:
            cluster_counts = clustered_df['cluster'].value_counts()
            # Exclude noise cluster (-1) from counts for visualization
            if -1 in cluster_counts.index:
                noise_count = cluster_counts[-1]
                cluster_counts = cluster_counts.drop(index=-1)
                st.info(f"💡 {noise_count} posts were identified as noise (Cluster -1) and are excluded from cluster visualization but still included in the network graph if they are influencers.")

            if not cluster_counts.empty:
                st.markdown("### 🤖 Detected Coordination Clusters")
                st.write("This chart visualizes the sizes of detected clusters, where each cluster represents a group of coordinated texts.")
                fig_clust = px.bar(
                    cluster_counts,
                    title="Cluster Sizes",
                    labels={'value': 'Member Count', 'index': 'Cluster ID'},
                    color=cluster_counts.index.astype(str),
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_clust, use_container_width=True)
                st.write("This table shows the influencers, their posts, timestamps, and their assigned cluster IDs.")
                st.dataframe(clustered_df[['Influencer', 'text', 'Timestamp', 'cluster']])
            else:
                st.info("No significant clusters detected (all posts might be noise or too few posts for clustering).")
        else:
            st.info("No data available for clustering.")

    except Exception as e:
        st.warning(f"⚠️ Clustering analysis failed: {e}")

    st.markdown("### 🕸️ User Interaction Network")
    try:
        graph_df = clustered_df if 'clustered_df' in locals() and not clustered_df.empty else filtered_df

        if graph_df.empty or graph_df['Influencer'].dropna().empty:
            st.info("No valid influencer data to build the network graph.")
        else:
            G, pos, cluster_map = cached_network_graph(graph_df)

            if not G.nodes():
                st.info("No nodes to display in the network graph. This might be due to filtered data or issues in graph creation.")
            else:
                edge_trace = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'))

                node_colors = [cluster_map.get(node, -2) for node in G.nodes()] # Use -2 for nodes not in any detected cluster
                unique_clusters = sorted(list(set(node_colors)))

                # Assign colors based on unique_clusters to ensure consistent coloring
                # and handle the '-1' noise cluster if present
                color_map = {c: i for i, c in enumerate(unique_clusters)}
                mapped_node_colors = [color_map[c] for c in node_colors]

                # If only one cluster (0 or -1), use a single color, otherwise use colorscale
                if len(unique_clusters) == 1:
                    marker_colorscale = px.colors.qualitative.Plotly # Or any single color scale
                    node_color_vals = [mapped_node_colors[0]] * len(mapped_node_colors)
                    colorbar_dict = None
                else:
                    # Filter out -1 for discrete color mapping if it's not desired in legend/colorbar
                    display_clusters = [c for c in unique_clusters if c != -1]
                    color_palette = px.colors.qualitative.Set3 * 5 # Extend palette if many clusters
                    marker_colors = [color_palette[color_map[c]] for c in unique_clusters]
                    
                    node_color_vals = mapped_node_colors # Use mapped indices for color
                    
                    # Custom colorbar/legend if needed, for simplicity let's rely on Plotly's default discrete color handling
                    colorbar_dict = dict(
                        title="Clusters",
                        tickvals=list(color_map.values()),
                        ticktext=[str(c) for c in unique_clusters]
                    )

                node_trace = go.Scatter(
                    x=[pos[node][0] for node in G.nodes()],
                    y=[pos[node][1] for node in G.nodes()],
                    text=[f"Influencer: {node}<br>Cluster: {cluster_map.get(node, 'N/A')}" for node in G.nodes()],
                    mode='markers+text',
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=node_color_vals,
                        colorscale='Plotly' if len(unique_clusters) > 1 else None, # Use Plotly default for discrete coloring
                        cmin=0, cmax=len(unique_clusters)-1, # Define color range for discrete values
                        line=dict(width=2, color='darkblue'),
                        colorbar=colorbar_dict if len(unique_clusters) > 1 else None,
                        colors=marker_colors if len(unique_clusters) > 1 else marker_colorscale # This line might be redundant with color=node_color_vals and colorscale
                    ),
                    hoverinfo='text'
                )

                st.write("This interactive graph visualizes the network of influencers, with nodes representing influencers and edges indicating interactions or shared narratives. Nodes are colored by their detected cluster.")
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
                st.write("This chart identifies influencers who appear in 3 or more coordinated messages, potentially indicating high-risk accounts.")
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
