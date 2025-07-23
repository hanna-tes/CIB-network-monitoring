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
        return "X"
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
    file_name = "TogoJULYData - Sheet1.csv" # Ensure this file is in the same directory as the script
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

    # Convert all column names to a consistent format (e.g., strip spaces, store original for display)
    # No need to keep original columns, just strip for internal use
    df.columns = df.columns.str.strip() # Clean column names for internal processing

    # Helper to find column from candidates (case-insensitive)
    def find_column(df_local, user_col, candidates):
        if user_col and user_col in df_local.columns and not df_local[user_col].astype(str).str.strip().eq('').all():
            return user_col
        
        # Try finding a candidate column (case-insensitive check)
        df_columns_lower = [col.lower() for col in df_local.columns]
        for candidate in candidates:
            if candidate.lower() in df_columns_lower:
                # Get the actual column name with its original casing
                actual_col = df_local.columns[df_columns_lower.index(candidate.lower())]
                if not df_local[actual_col].astype(str).str.strip().eq('').all():
                    return actual_col
        return None

    # --- Create 'text' column ---
    text_candidates_fallback = ['text', 'Hit Sentence', 'Opening Text', 'Headline', 'message', 'title', 'content', 'description', 'Body', 'FullText']
    chosen_text_col = find_column(df, user_text_col, text_candidates_fallback)
    if chosen_text_col:
        df['text'] = df[chosen_text_col].astype(str).replace('nan', np.nan).fillna('')
    else:
        st.warning(f"⚠️ No suitable text column found. Tried '{user_text_col}' and {text_candidates_fallback}. 'text' column might be empty.")
        df['text'] = ""

    df['text'] = df['text'].astype(str).replace('nan', np.nan)
    df = df.dropna(subset=['text']).reset_index(drop=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)

    # --- Populate 'Influencer' column ---
    influencer_candidates_fallback = [
        'Influencer', 'author', 'username', 'user', 'authorMeta/name', 'creator', 'authorname', 'Source', 'media_name'
    ]
    chosen_influencer_col = find_column(df, user_influencer_col, influencer_candidates_fallback)

    if chosen_influencer_col:
        df['Influencer'] = df[chosen_influencer_col].astype(str).replace('nan', np.nan).fillna('Unknown_User')
    else:
        st.warning(f"⚠️ No suitable Influencer column found. Tried '{user_influencer_col}' and {influencer_candidates_fallback}. Falling back to 'Outlet' if available, otherwise 'Unknown_User'.")
        # Fallback to Outlet if no direct influencer column is found
        outlet_candidates_fallback = ['media_name', 'channeltitle', 'source']
        chosen_outlet_col = find_column(df, user_outlet_col, outlet_candidates_fallback)
        if chosen_outlet_col:
            df['Influencer'] = df[chosen_outlet_col].astype(str).replace('nan', np.nan).fillna('Unknown_User')
            st.info(f"Using '{chosen_outlet_col}' as 'Influencer' column.")
        else:
            df['Influencer'] = "Unknown_User"
            st.warning("⚠️ No suitable Outlet column found to fall back on for Influencer. Influencer column set to 'Unknown_User'.")
    df['Influencer'] = df['Influencer'].astype(str).replace('nan', np.nan).fillna('Unknown_User')

    # --- Timestamp Parsing ---
    df['Timestamp'] = pd.NaT # Initialize with Not a Time
    date_candidates_fallback = ['Date', 'createTimeISO', 'published_date', 'pubDate', 'created_at', 'Alternate Date Format', 'publish_date']
    chosen_timestamp_col = find_column(df, user_timestamp_col, date_candidates_fallback)

    if chosen_timestamp_col:
        df['Timestamp'] = pd.to_datetime(df[chosen_timestamp_col], errors='coerce')
    else:
        st.warning(f"⚠️ No suitable Timestamp column found. Tried '{user_timestamp_col}' and {date_candidates_fallback}. Timestamp column might be incomplete.")

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

        parsed = pd.to_datetime(timestamp, errors='coerce')
        if pd.notna(parsed): return parsed

        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce')
                if pd.notna(parsed): return parsed
            except (ValueError, TypeError): continue
        return pd.NaT

    # Apply robust parsing to any remaining NaT values
    if df['Timestamp'].isna().any():
        if chosen_timestamp_col:
            # Re-attempt parsing for NaNs in the chosen column
            df.loc[df['Timestamp'].isna(), 'Timestamp'] = df.loc[df['Timestamp'].isna(), chosen_timestamp_col].apply(parse_timestamp_robust)
        else:
            # If no column was initially chosen, iterate through fallbacks for NaNs
            for col_name in date_candidates_fallback:
                # Find the actual column name (case-insensitive)
                df_columns_lower = [col.lower() for col in df.columns]
                if col_name.lower() in df_columns_lower:
                    actual_fallback_col = df.columns[df_columns_lower.index(col_name.lower())]
                    df.loc[df['Timestamp'].isna(), 'Timestamp'] = df.loc[df['Timestamp'].isna(), actual_fallback_col].apply(parse_timestamp_robust)
                    if not df['Timestamp'].isna().all():
                        break

    def localize_to_utc(dt):
        if pd.isna(dt): return dt
        if dt.tzinfo is None: return dt.tz_localize('UTC')
        else: return dt.tz_convert('UTC')

    df['Timestamp'] = df['Timestamp'].apply(localize_to_utc)
    df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)

    # --- Create 'URL' column ---
    url_candidates_fallback = ['URL', 'url', 'webVideoUrl', 'link', 'post_url', 'media_url']
    chosen_url_col = find_column(df, user_url_col, url_candidates_fallback)

    if chosen_url_col:
        df['URL'] = df[chosen_url_col].astype(str).replace('nan', np.nan).fillna(np.nan)
    else:
        df['URL'] = np.nan
        st.sidebar.warning(f"⚠️ No suitable URL column found. Tried '{user_url_col}' and {url_candidates_fallback}. Platform detection will be limited.")

    # --- Create 'Platform' from URL or existing 'Platform' column ---
    if 'Platform' in df.columns and not df['Platform'].empty and df['Platform'].notna().any():
        # If a 'Platform' column already exists in the original data, use it.
        # Ensure its values are reasonable (e.g., string, not all NaN/empty)
        if df['Platform'].astype(str).str.strip().eq('').all(): # if existing platform column is empty
             if 'URL' in df.columns and not df['URL'].empty and df['URL'].notna().any():
                df['Platform'] = df['URL'].apply(infer_platform_from_url)
             else:
                df['Platform'] = "Unknown"
        # Otherwise, assume existing Platform column is good
    elif 'URL' in df.columns and not df['URL'].empty and df['URL'].notna().any():
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
    including URLs and Platforms for context.
    """
    clean_df = df[['original_text', 'Influencer', 'Timestamp', 'URL', 'Platform']].copy()
    clean_df['original_text'] = clean_df['original_text'].astype(str)
    clean_df = clean_df.dropna(subset=['original_text', 'Influencer', 'Timestamp', 'Platform'])
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
        
        platforms_involved = sorted(list(set([row1['Platform'], row2['Platform']])))
        platforms_involved_str = ", ".join(platforms_involved)

        similar_pairs.append({
            'text1': row1['original_text'],
            'influencer1': row1['Influencer'],
            'platform1': row1['Platform'], # Added platform1
            'time1': row1['Timestamp'],
            'url1': row1['URL'],
            'text2': row2['original_text'],
            'influencer2': row2['Influencer'],
            'platform2': row2['Platform'], # Added platform2
            'time2': row2['Timestamp'],
            'url2': row2['URL'],
            'similarity': round(sim_matrix[i, j], 3),
            'shared_narrative': narrative_snippet,
            'platforms_involved': platforms_involved_str # New column for summary
        })
    return pd.DataFrame(similar_pairs)

# --- Clustering and Graph Building Functions ---
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
    grouped = df.groupby('cluster')
    for cluster_id, group in grouped:
        if cluster_id == -1 or len(group) < 2: # -1 is noise, or too few members
            continue
        users = group['Influencer'].dropna().unique().tolist()
        for u1, u2 in combinations(users, 2):
            if G.has_edge(u1, u2):
                G[u1][u2]['weight'] += 1
            else:
                G.add_edge(u1, u2, weight=1)

    all_influencers = df['Influencer'].dropna().unique().tolist()
    # Also capture their primary platform for node attributes
    influencer_platform_map = df.groupby('Influencer')['Platform'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').to_dict()

    for inf in all_influencers:
        if inf not in G.nodes():
            G.add_node(inf)
        G.nodes[inf]['platform'] = influencer_platform_map.get(inf, 'Unknown')


    pos = nx.spring_layout(G, seed=42, k=0.1, iterations=50)

    cluster_map = df.set_index('Influencer')['cluster'].to_dict()
    final_cluster_map = {node: cluster_map.get(node, 0) for node in G.nodes()}

    return G, pos, final_cluster_map


# --- Cached Expensive Functions ---
@st.cache_data(show_spinner="🔍 Computing textual similarities...")
def cached_similarity_analysis(_df, threshold=0.85):
    return find_textual_similarities(_df, threshold)

@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df):
    """
    Performstext clustering using the integrated DBSCAN clustering function.
    """
    return cluster_texts(_df)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df):
    """
    Builds a user interaction network graph using the integrated function.
    """
    return build_user_interaction_graph(_df)

# --- Data Source Selection ---
st.sidebar.header("📥 Data Source")
data_source_option = st.sidebar.radio(
    "Choose data source:",
    ("Use Default Data", "Upload CSV")
)

df = pd.DataFrame()

if data_source_option == "Use Default Data":
    df = load_default_dataset()
elif data_source_option == "Upload CSV":
    uploaded_files = st.sidebar.file_uploader("Upload your CSV file(s)", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        if len(uploaded_files) > 1:
            st.sidebar.warning("You uploaded multiple files. Please ensure critical columns (like text, influencer, timestamp, URL) have consistent names across all files for accurate mapping and analysis.")
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
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

# Exit if no data after selection
if df is None or df.empty:
    st.warning("No data available. Please select a data source and ensure it's valid.")
    st.stop()

# Get columns for selectboxes
all_columns = df.columns.tolist()
# Add a default selection option at the beginning
column_selection_options = ["-- Select Column --"] + all_columns

# Function to get default index for selectbox (case-insensitive)
def get_default_index(col_name, options):
    try:
        lower_options = [opt.lower() for opt in options]
        # First, try exact match, then case-insensitive match
        if col_name in options:
            return options.index(col_name)
        
        if col_name.lower() in lower_options:
            return lower_options.index(col_name.lower()) # Return index of the first match
    except ValueError:
        pass
    return 0 # Default to "-- Select Column --"

# --- Flexible Column Mapping Input ---
st.sidebar.header("⚙️ Column Mappings")
st.sidebar.markdown("Please select the correct columns from your data:")

user_text_col = st.sidebar.selectbox(
    "Main Text Column",
    options=column_selection_options,
    index=get_default_index("text", column_selection_options),
    help="Select the column containing the main text of the posts (e.g., 'message', 'content', 'FullText')."
)
user_influencer_col = st.sidebar.selectbox(
    "Influencer/Author Column",
    options=column_selection_options,
    index=get_default_index("Influencer", column_selection_options),
    help="Select the column identifying the influencer or author (e.g., 'username', 'author', 'Source')."
)
user_timestamp_col = st.sidebar.selectbox(
    "Timestamp Column",
    options=column_selection_options,
    index=get_default_index("Timestamp", column_selection_options),
    help="Select the column containing the date and time of the post (e.g., 'Date', 'published_date', 'created_at')."
)
user_url_col = st.sidebar.selectbox(
    "URL Column",
    options=column_selection_options,
    index=get_default_index("URL", column_selection_options),
    help="Select the column with the URL of the post (e.g., 'link', 'post_url', 'webVideoUrl'). This is used to infer the platform."
)
user_outlet_col = st.sidebar.selectbox(
    "Media Outlet/Channel Column (Optional)",
    options=column_selection_options,
    index=get_default_index("Outlet", column_selection_options),
    help="Select the column for media outlet or channel. This can be used as a fallback for Influencer if no specific influencer column is found."
)

# Warn if default selection is still present for critical columns
if "-- Select Column --" in [user_text_col, user_influencer_col, user_timestamp_col]:
    st.sidebar.warning("Please ensure all required column mappings are selected from the dropdowns.")
    st.stop()


# --- Preprocess ---
with st.spinner("⏳ Preprocessing data..."):
    df = preprocess_data(df, user_text_col, user_influencer_col, user_timestamp_col, user_url_col, user_outlet_col)

# Exit if no data after preprocessing
if df.empty:
    st.warning("No valid data available after preprocessing. Please check your data file and column mappings.")
    st.stop()

# --- Sidebar Filters (Global Filters) ---
st.sidebar.header("🔍 Global Filters (Apply to all tabs)")

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

available_platforms_global = df['Platform'].dropna().astype(str).unique().tolist()
platforms_global = st.sidebar.multiselect(
    "Platforms",
    options=available_platforms_global,
    default=available_platforms_global
)

# Apply global filters
filtered_df_global = df[
    (df['Timestamp'] >= start_dt) &
    (df['Timestamp'] <= end_dt) &
    (df['Platform'].isin(platforms_global))
].copy()

if filtered_df_global.empty:
    st.warning("No data matches the selected global filters. Please adjust the date range or platforms.")
    st.stop()

# Export button
st.sidebar.markdown("### 📄 Export Results")
@st.cache_data
def convert_df(data):
    return data.to_csv(index=False).encode('utf-8')

csv_data = convert_df(filtered_df_global)
st.sidebar.download_button("Download Filtered Data", csv_data, "filtered_data.csv", "text/csv")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Summary Statistics")

    st.markdown("### 🔬 Preprocessed Data Sample (for debugging column values)")
    st.write("Check the values in 'Influencer', 'Platform', and 'URL' columns below to ensure they are correctly identified after preprocessing.")
    st.dataframe(df[['Influencer', 'Platform', 'URL']].head(10))
    st.markdown("---")

    if not filtered_df_global.empty:
        st.write("This chart shows the top 10 influencers by the number of posts in the filtered dataset.")
        top_influencers = filtered_df_global['Influencer'].value_counts().head(10)
        fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'Influencer'})
        st.plotly_chart(fig_src, use_container_width=True)

        if 'Platform' in filtered_df_global.columns and not filtered_df_global['Platform'].empty:
            st.write("This chart displays the distribution of posts across all identified social media and media platforms in the dataset.")
            all_platforms_counts = filtered_df_global['Platform'].value_counts()
            fig_platform = px.bar(all_platforms_counts, title="Post Distribution by Platform", labels={'value': 'Posts', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True)
        else:
            st.info("No 'Platform' column found or no data for platforms. This typically happens if no URLs are present in the data.")

        if 'Outlet' in filtered_df_global.columns and not filtered_df_global['Outlet'].empty:
            st.write("This chart illustrates the top 10 media outlets or channels where content was published.")
            top_outlets = filtered_df_global['Outlet'].value_counts().head(10)
            fig_outlet = px.bar(top_outlets, title="Top 10 Media Outlets/Channels", labels={'value': 'Posts', 'index': 'Outlet'})
            st.plotly_chart(fig_outlet, use_container_width=True)
        # Note: 'Channel' is no longer a primary target, as 'Outlet' handles this via mapping/fallbacks.
        # This elif block is kept for backward compatibility if 'Channel' exists and 'Outlet' does not.
        elif 'Channel' in filtered_df_global.columns and not filtered_df_global['Channel'].empty:
            st.write("This chart illustrates the top 10 channels where content was published.")
            top_channels = filtered_df_global['Channel'].value_counts().head(10)
            fig_chan = px.bar(top_channels, title="Top 10 Channels", labels={'value': 'Posts', 'index': 'Channel'})
            st.plotly_chart(fig_chan, use_container_width=True)

        # Conditional display for Top 10 Hashtags - only for non-Media platforms
        if 'Platform' in filtered_df_global.columns and not filtered_df_global['Platform'].empty:
            social_media_df = filtered_df_global[filtered_df_global['Platform'] != 'Media'].copy()
            
            if social_media_df.empty:
                st.info("Hashtag analysis skipped: No social media (non-'Media') content found in the filtered data.")
            else:
                if 'text' in social_media_df.columns and not social_media_df['text'].empty:
                    st.write("This chart highlights the top 10 most frequently used hashtags, focusing on social media content where hashtags are typically relevant.")
                    social_media_df['hashtags'] = social_media_df['text'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])

                    all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list if tags_list]

                    if all_hashtags:
                        hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                        fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                        st.plotly_chart(fig_ht, use_container_width=True)
                    else:
                        st.info("No hashtags found in the social media 'text' column.")
                else:
                    st.info("No 'text' column found or it's empty to extract hashtags from social media content.")
        else:
            st.info("Cannot determine platform for hashtag analysis (no 'Platform' column or empty).")


        st.write("This area chart visualizes the daily volume of posts over the selected date range.")
        time_series = filtered_df_global.set_index('Timestamp').resample('D').size()
        fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'Timestamp': 'Date'})
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
    """)
    
    st.markdown("---")
    st.subheader("Filters for Analysis (Tab 2 Only)")
    available_platforms_analysis = filtered_df_global['Platform'].dropna().astype(str).unique().tolist()
    platforms_analysis = st.multiselect(
        "Platforms to include in Similarity Analysis:",
        options=available_platforms_analysis,
        default=available_platforms_analysis,
        key="platforms_analysis_tab2" # Unique key for this widget
    )
    
    analysis_df_filtered_by_platform = filtered_df_global[filtered_df_global['Platform'].isin(platforms_analysis)].copy()

    MAX_ROWS_SIMILARITY = st.slider("Max posts to analyze for similarity (for performance)", 100, 1000, 300, key="max_rows_similarity")
    
    # Ensure original_text column is present and valid
    if 'original_text' not in analysis_df_filtered_by_platform.columns:
        analysis_df_filtered_by_platform['original_text'] = analysis_df_filtered_by_platform['text'].apply(extract_original_text)

    # Create a fresh copy of filtered_df for analysis_df to ensure cache invalidation
    analysis_df = analysis_df_filtered_by_platform[analysis_df_filtered_by_platform['original_text'].astype(str).str.strip() != ""].head(MAX_ROWS_SIMILARITY).copy()

    if analysis_df.empty:
        st.info("No valid text data available for similarity analysis after applying filters and row limit.")
    else:
        with st.spinner(f"🔍 Finding coordinated narratives among {len(analysis_df)} posts..."):
            sim_df = cached_similarity_analysis(analysis_df, threshold=0.85)

        if not sim_df.empty:
            st.success(f"✅ Found {len(sim_df)} similar pairs.")
            narrative_summary = sim_df.groupby('shared_narrative').agg(
                share_count=('similarity', 'count'),
                influencers_involved=('influencer1', lambda x: ", ".join(x.astype(str).unique()[:5]) + ("..." if len(x.unique()) > 5 else "")),
                platforms_involved=('platforms_involved', lambda x: ", ".join(sorted(list(set([p.strip() for sublist in x.tolist() for p in sublist.split(',')])))))
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
            st.markdown("### 🔄 Full Similarity Pairs")
            st.write("This table lists all detected pairs of similar texts, along with their influencers, platforms, timestamps, similarity scores, and links to the original posts for verification.")

            display_sim_df = sim_df[['text1', 'influencer1', 'platform1', 'time1', 'url1', 'text2', 'influencer2', 'platform2', 'time2', 'url2', 'similarity']].copy()
            display_sim_df['url1'] = display_sim_df['url1'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) else '')
            display_sim_df['url2'] = display_sim_df['url2'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) else '')

            st.markdown(display_sim_df.to_html(escape=False), unsafe_allow_html=True)

        else:
            st.info("No significant similarities found above threshold.")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")

    st.markdown("---")
    st.subheader("Filters for Analysis (Tab 3 Only)")
    available_platforms_network = filtered_df_global['Platform'].dropna().astype(str).unique().tolist()
    platforms_network = st.multiselect(
        "Platforms to include in Network & Risk Analysis:",
        options=available_platforms_network,
        default=available_platforms_network,
        key="platforms_network_tab3" # Unique key for this widget
    )
    
    network_df_filtered_by_platform = filtered_df_global[filtered_df_global['Platform'].isin(platforms_network)].copy()

    max_influencers_graph = st.slider(
        "Max Influencers for Network Graph (for performance)",
        min_value=10, max_value=200, value=50, step=10,
        help="Limit the number of influencers displayed in the network graph to improve performance.",
        key="max_influencers_graph"
    )

    try:
        # Ensure original_text column is present and valid
        if 'original_text' not in network_df_filtered_by_platform.columns:
            network_df_filtered_by_platform['original_text'] = network_df_filtered_by_platform['text'].apply(extract_original_text)

        # Create a fresh copy of filtered_df for df_for_clustering to ensure cache invalidation
        df_for_clustering = network_df_filtered_by_platform[network_df_filtered_by_platform['text'].astype(str).str.strip() != ""].copy()
        
        if df_for_clustering.empty:
            st.info("No valid text data for clustering analysis.")
            clustered_df = pd.DataFrame()
        else:
            clustered_df = cached_clustering(df_for_clustering)

        if not clustered_df.empty:
            cluster_counts = clustered_df['cluster'].value_counts()
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
                st.dataframe(clustered_df[['Influencer', 'Platform', 'text', 'Timestamp', 'cluster']])
            else:
                st.info("No significant clusters detected (all posts might be noise or too few posts for clustering).")
        else:
            st.info("No data available for clustering.")

    except Exception as e:
        st.warning(f"⚠️ Clustering analysis failed: {e}")

    st.markdown("### 🕸️ User Interaction Network")
    st.markdown("""
        This interactive graph shows how different **influencers** (represented by **nodes** or circles) are connected.
        A line (or **edge**) between two influencers means they have shared similar content or been part of the same coordinated narrative (cluster).
        
        **How to interpret the colors:**
        The colors of the nodes are assigned automatically to visually group influencers that belong to the same detected cluster.
        For example, all influencers within the 'blue' group are part of one coordinated cluster, while those in the 'green' group belong to a different one.
        The specific meaning of each color is not fixed (e.g., 'red' doesn't always mean the same thing across different analyses), but its purpose is to help you quickly see which influencers are working together on similar themes.
    """)
    try:
        # Ensure graph_df is always a fresh copy of the relevant DataFrame
        graph_df = clustered_df.copy() if 'clustered_df' in locals() and not clustered_df.empty else network_df_filtered_by_platform.copy()

        if graph_df.empty or graph_df['Influencer'].dropna().empty:
            st.info("No valid influencer data to build the network graph.")
        else:
            # Prioritize influencers in clusters, then by post count, up to max_influencers_graph
            clustered_influencers = graph_df[graph_df['cluster'] != -1]['Influencer'].value_counts().index.tolist()
            all_influencers_by_post = graph_df['Influencer'].value_counts().index.tolist()

            selected_influencers = []
            seen_influencers = set()

            # Add clustered influencers first
            for inf in clustered_influencers:
                if inf not in seen_influencers and len(selected_influencers) < max_influencers_graph:
                    selected_influencers.append(inf)
                    seen_influencers.add(inf)

            # Fill up with other high-post-count influencers if space remains
            for inf in all_influencers_by_post:
                if inf not in seen_influencers and len(selected_influencers) < max_influencers_graph:
                    selected_influencers.append(inf)
                    seen_influencers.add(inf)
            
            # Filter graph_df for only the selected influencers
            graph_df_subset = graph_df[graph_df['Influencer'].isin(selected_influencers)].copy()

            if graph_df_subset.empty:
                st.info("No data for selected influencers to build the network graph. Try increasing the 'Max Influencers for Network Graph' slider.")
            else:
                G, pos, cluster_map = cached_network_graph(graph_df_subset)

                if not G.nodes():
                    st.info("No nodes to display in the network graph. This might be due to filtered data or issues in graph creation.")
                else:
                    edge_trace = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_trace.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'))

                    node_colors = [cluster_map.get(node, -2) for node in G.nodes()] # Use -2 for non-clustered/noise
                    unique_clusters = sorted(list(set(node_colors)))

                    color_map = {c: i for i, c in enumerate(unique_clusters)}
                    mapped_node_colors = [color_map[c] for c in node_colors]

                    if len(unique_clusters) == 1:
                        marker_colorscale = px.colors.qualitative.Plotly[0]
                        node_color_vals = [marker_colorscale] * len(mapped_node_colors)
                        colorbar_dict = None
                    else:
                        color_palette = px.colors.qualitative.Set3
                        extended_color_palette = color_palette * ((len(unique_clusters) // len(color_palette)) + 1)

                        marker_colors = [extended_color_palette[color_map[c]] for c in unique_clusters]

                        node_color_vals = [extended_color_palette[color_map[c]] for c in node_colors]

                        colorbar_dict = dict(
                            title="Clusters",
                            tickvals=[color_map[c] for c in unique_clusters],
                            ticktext=[str(c) if c != -2 else 'Not Clustered' for c in unique_clusters], # Better label for -2
                            x=1.02,
                            xanchor="left",
                            len=0.7
                        )

                    node_trace = go.Scatter(
                        x=[pos[node][0] for node in G.nodes()],
                        y=[pos[node][1] for node in G.nodes()],
                        text=[f"Influencer: {node}<br>Cluster: {cluster_map.get(node, 'N/A')}<br>Platform: {G.nodes[node].get('platform', 'N/A')}" for node in G.nodes()], # Added platform
                        mode='markers+text',
                        textposition="top center",
                        marker=dict(
                            size=12,
                            color=node_color_vals,
                            line=dict(width=2, color='darkblue'),
                            colorbar=colorbar_dict
                        ),
                        hoverinfo='text'
                    )

                    st.write("This interactive graph visualizes the network of influencers, with nodes representing influencers and edges indicating interactions or shared narratives. Nodes are colored by their detected cluster.")
                    if len(selected_influencers) < len(seen_influencers):
                         st.info(f"Displaying a subset of {len(G.nodes())} influencers in the network graph based on your filter and the 'Max Influencers for Network Graph' setting. Adjust the slider to include more influencers.")
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
    st.markdown("""
        **High-risk influencers** are those who frequently participate in **coordinated messages**.
        This chart highlights influencers who appear in 3 or more similar messages.
        A high count here could indicate that an influencer is a central figure in spreading specific narratives or is part of a concentrated effort.
    """)
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
