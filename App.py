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
def preprocess_data(df, user_text_col, user_influencer_col, user_timestamp_col, user_url_col, user_outlet_col):
    """
    Preprocesses the DataFrame: maps columns, creates 'text' column, cleans text,
    parses and localizes timestamps, and infers platform.
    Uses user-defined column names first, then falls back to predefined candidates.
    """
    # 1. Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert all column names to a consistent format (e.g., lowercased, no spaces) for robust matching
    df.columns = df.columns.str.strip() # Remove leading/trailing spaces from column names

    # --- Create 'text' column using user-defined name or fallbacks ---
    df['text'] = ''
    found_primary_text_col = False
    
    # Try user-defined text column first
    if user_text_col and user_text_col in df.columns:
        df['text'] = df[user_text_col].astype(str).replace('nan', np.nan).fillna('')
        if not df['text'].astype(str).str.strip().eq('').all():
            found_primary_text_col = True
    
    if not found_primary_text_col:
        text_candidates_fallback = ['text', 'Hit Sentence', 'Opening Text', 'Headline', 'message', 'title', 'content', 'description', 'Body', 'FullText']
        for col_name in text_candidates_fallback:
            if col_name in df.columns:
                df['text'] = df[col_name].astype(str).replace('nan', np.nan).fillna('')
                if not df['text'].astype(str).str.strip().eq('').all():
                    found_primary_text_col = True
                    break
    
    if not found_primary_text_col:
        st.warning("⚠️ No suitable text column found. 'text' column might be empty.")
        df['text'] = ""

    df['text'] = df['text'].astype(str).replace('nan', np.nan)
    df = df.dropna(subset=['text']).reset_index(drop=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)

    # --- Populate 'Influencer' column using user-defined name or fallbacks ---
    df['Influencer'] = "Unknown_User"

    if user_influencer_col and user_influencer_col in df.columns:
        df['Influencer'] = df[user_influencer_col].astype(str).replace('nan', np.nan).fillna('Unknown_User')
    else:
        influencer_candidates_fallback = [
            'Influencer', 'author', 'username', 'user', 'authorMeta/name', 'creator', 'authorname', 'Source'
        ]
        for col_name in influencer_candidates_fallback:
            if col_name in df.columns and (df['Influencer'] == "Unknown_User").any(): # Only update if still 'Unknown_User'
                df['Influencer'] = df['Influencer'].mask(
                    (df['Influencer'] == "Unknown_User") | df['Influencer'].astype(str).str.strip().eq(''),
                    df[col_name].astype(str).replace('nan', np.nan).fillna('Unknown_User')
                )
    
    # Last resort: Use Outlet as Influencer if no other influencer found
    if user_outlet_col and user_outlet_col in df.columns:
        df['Outlet'] = df[user_outlet_col].astype(str).replace('nan', np.nan).fillna('Unknown_Outlet')
        df['Influencer'] = df['Influencer'].mask(
            (df['Influencer'] == "Unknown_User") | df['Influencer'].astype(str).str.strip().eq(''),
            df['Outlet'].astype(str).replace('nan', np.nan).fillna('Unknown_User')
        )
    else: # Fallback for outlet if user didn't specify
        outlet_candidates_fallback = ['media_name', 'channeltitle', 'source']
        for col_name in outlet_candidates_fallback:
            if col_name in df.columns:
                df['Outlet'] = df[col_name].astype(str).replace('nan', np.nan).fillna('Unknown_Outlet')
                df['Influencer'] = df['Influencer'].mask(
                    (df['Influencer'] == "Unknown_User") | df['Influencer'].astype(str).str.strip().eq(''),
                    df['Outlet'].astype(str).replace('nan', np.nan).fillna('Unknown_User')
                )
                break


    df['Influencer'] = df['Influencer'].astype(str).replace('nan', np.nan).fillna('Unknown_User')
    
    # --- Timestamp Parsing using user-defined name or fallbacks ---
    df['Timestamp'] = pd.NaT # Initialize with Not a Time
    
    if user_timestamp_col and user_timestamp_col in df.columns:
        # st.info(f"Attempting to parse timestamp using user-defined column: '{user_timestamp_col}'") # Commented out
        df['Timestamp'] = pd.to_datetime(df[user_timestamp_col], errors='coerce')
    
    # Fallback to predefined timestamp columns if user-defined failed or not provided
    if df['Timestamp'].isna().all(): # If all values are NaT after user-defined attempt
        st.warning("⚠️ User-defined Timestamp column failed or not found. Falling back to other date candidates.")
        date_candidates_fallback = ['Date', 'createTimeISO', 'published_date', 'pubDate', 'created_at', 'Alternate Date Format']
        for col_name in date_candidates_fallback:
            if col_name in df.columns:
                df['Timestamp'] = pd.to_datetime(df[col_name], errors='coerce')
                if not df['Timestamp'].isna().all():
                    # st.info(f"Successfully parsed timestamp using fallback column: '{col_name}'") # Commented out
                    break
    
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
        if isinstance(timestamp, pd.Timestamp): # Already a timestamp
            return timestamp
        
        # Try generic conversion first
        parsed = pd.to_datetime(timestamp, errors='coerce')
        if pd.notna(parsed): return parsed

        # Then try specific formats
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce')
                if pd.notna(parsed): return parsed
            except (ValueError, TypeError): continue
        return pd.NaT
    
    # Apply robust parsing to any remaining NaT values or if initial parsing failed
    if df['Timestamp'].isna().any():
        # Iterate over rows and apply parsing if Timestamp is still NaT
        # This approach can be slow for very large datasets, but more robust for mixed formats
        for idx, row in df[df['Timestamp'].isna()].iterrows():
            if user_timestamp_col in row and pd.notna(row[user_timestamp_col]):
                df.loc[idx, 'Timestamp'] = parse_timestamp_robust(row[user_timestamp_col])
            else: # Try fallback candidates if user_timestamp_col wasn't available or was empty
                for col_name in date_candidates_fallback:
                    if col_name in row and pd.notna(row[col_name]):
                        df.loc[idx, 'Timestamp'] = parse_timestamp_robust(row[col_name])
                        if pd.notna(df.loc[idx, 'Timestamp']):
                            break


    def localize_to_utc(dt):
        if pd.isna(dt): return dt
        if dt.tzinfo is None: return dt.tz_localize('UTC')
        else: return dt.tz_convert('UTC')

    df['Timestamp'] = df['Timestamp'].apply(localize_to_utc)
    df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)

    # --- Create 'URL' column using user-defined name or fallbacks ---
    df['URL'] = np.nan # Initialize as NaN

    if user_url_col and user_url_col in df.columns:
        df['URL'] = df[user_url_col].astype(str).replace('nan', np.nan).fillna(np.nan)
    else:
        url_candidates_fallback = ['URL', 'url', 'webVideoUrl', 'link', 'post_url']
        for col_name in url_candidates_fallback:
            if col_name in df.columns:
                df['URL'] = df[col_name].astype(str).replace('nan', np.nan).fillna(np.nan)
                break
    
    # --- Create 'Platform' from URL ---
    if 'URL' in df.columns and not df['URL'].empty:
        df['Platform'] = df['URL'].apply(infer_platform_from_url)
    else:
        df['Platform'] = "Unknown"
        st.sidebar.warning("⚠️ No URL column found or URL column is empty → all platforms marked as 'Unknown'")

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
    # Use 'Influencer' as the key for the cluster map
    cluster_map = df.set_index('Influencer')['cluster'].to_dict()
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

# --- Flexible Column Mapping Input ---
st.sidebar.header("⚙️ Column Mappings")
st.sidebar.markdown("Specify the column names from your data for key fields.")

user_text_col = st.sidebar.text_input("Main Text Column", value="text", help="e.g., 'message', 'content', 'FullText', 'text'")
user_influencer_col = st.sidebar.text_input("Influencer/Author Column", value="Influencer", help="e.g., 'username', 'author', 'Source'")
user_timestamp_col = st.sidebar.text_input("Timestamp Column", value="Timestamp", help="e.g., 'Date', 'published_date', 'created_at'")
user_url_col = st.sidebar.text_input("URL Column", value="URL", help="e.g., 'link', 'post_url', 'url'")
user_outlet_col = st.sidebar.text_input("Media Outlet/Channel Column (Optional)", value="Outlet", help="e.g., 'media_name', 'channeltitle', 'source'")


# --- Preprocess ---
with st.spinner("⏳ Preprocessing data..."):
    df = preprocess_data(df, user_text_col, user_influencer_col, user_timestamp_col, user_url_col, user_outlet_col)

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

        if 'Channel' in filtered_df.columns: # 'Channel' is still mapped from 'channeltitle' or similar in default candidates
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
    
    # New slider for max influencers in graph
    max_influencers_graph = st.sidebar.slider(
        "Max Influencers for Network Graph", 
        min_value=10, max_value=200, value=50, step=10,
        help="Limit the number of influencers displayed in the network graph to improve performance."
    )

    try:
        if 'original_text' not in filtered_df.columns:
            filtered_df['original_text'] = filtered_df['text'].apply(extract_original_text)

        df_for_clustering = filtered_df[filtered_df['text'].astype(str).str.strip() != ""].copy()
        if df_for_clustering.empty:
            st.info("No valid text data for clustering analysis.")
            clustered_df = pd.DataFrame()
        else:
            clustered_df = cached_clustering(df_for_clustering)

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
            # Filter graph_df based on max_influencers_graph
            # Prioritize influencers in non-noise clusters, then top influencers by post count
            
            clustered_influencers = graph_df[graph_df['cluster'] != -1]['Influencer'].value_counts().index.tolist()
            top_influencers_by_post = graph_df['Influencer'].value_counts().index.tolist()
            
            # Combine and get unique influencers up to max_influencers_graph
            selected_influencers = []
            seen_influencers = set()
            
            for inf in clustered_influencers:
                if inf not in seen_influencers and len(selected_influencers) < max_influencers_graph:
                    selected_influencers.append(inf)
                    seen_influencers.add(inf)
            
            for inf in top_influencers_by_post:
                if inf not in seen_influencers and len(selected_influencers) < max_influencers_graph:
                    selected_influencers.append(inf)
                    seen_influencers.add(inf)
            
            # Filter the DataFrame to only include posts from selected influencers
            graph_df_subset = graph_df[graph_df['Influencer'].isin(selected_influencers)].copy()

            if graph_df_subset.empty:
                st.info("No data for selected influencers to build the network graph.")
            else:
                G, pos, cluster_map = cached_network_graph(graph_df_subset) # Pass the subset

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
                    color_map = {c: i for i, c in enumerate(unique_clusters)}
                    mapped_node_colors = [color_map[c] for c in node_colors]

                    if len(unique_clusters) == 1:
                        # If only one cluster, use a single consistent color and no colorbar
                        marker_colorscale = px.colors.qualitative.Plotly[0] # First color from Plotly palette
                        node_color_vals = [marker_colorscale] * len(mapped_node_colors)
                        colorbar_dict = None
                    else:
                        # For multiple clusters, use a discrete color scale
                        color_palette = px.colors.qualitative.Set3 # Base palette
                        # Ensure enough colors for all unique clusters
                        extended_color_palette = color_palette * ((len(unique_clusters) // len(color_palette)) + 1)
                        
                        marker_colors = [extended_color_palette[color_map[c]] for c in unique_clusters]
                        
                        node_color_vals = [extended_color_palette[color_map[c]] for c in node_colors]

                        colorbar_dict = dict(
                            title="Clusters",
                            tickvals=[color_map[c] for c in unique_clusters], # Tick values map to the numeric indices
                            ticktext=[str(c) for c in unique_clusters], # Tick text are the actual cluster IDs
                            x=1.02, # Position colorbar to the right
                            xanchor="left",
                            len=0.7 # Length of the colorbar
                        )

                    node_trace = go.Scatter(
                        x=[pos[node][0] for node in G.nodes()],
                        y=[pos[node][1] for node in G.nodes()],
                        text=[f"Influencer: {node}<br>Cluster: {cluster_map.get(node, 'N/A')}" for node in G.nodes()],
                        mode='markers+text',
                        textposition="top center",
                        marker=dict(
                            size=12,
                            color=node_color_vals, # Use the actual colors directly
                            # colorscale and cmin/cmax are typically for continuous scales
                            # For discrete colors, we provide a list of colors directly.
                            line=dict(width=2, color='darkblue'),
                            colorbar=colorbar_dict # Apply colorbar if multiple clusters
                        ),
                        hoverinfo='text'
                    )

                    st.write("This interactive graph visualizes the network of influencers, with nodes representing influencers and edges indicating interactions or shared narratives. Nodes are colored by their detected cluster.")
                    if len(selected_influencers) < len(seen_influencers):
                         st.info(f"Displaying a subset of {len(selected_influencers)} influencers in the network graph for performance. You can adjust the 'Max Influencers for Network Graph' slider in the sidebar.")
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
