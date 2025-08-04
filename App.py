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
    # Specifically target and remove RT @user: pattern at the beginning
    cleaned = re.sub(r'^(RT|rt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
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
    
    # Corrected line: filter the DataFrame directly, not just the 'text' column assignment
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

    df['Timestamp'] = df['Timestamp'].apply(lambda x: x.tz_localize('UTC') if pd.notna(x) and x.tzinfo is None else x.tz_convert('UTC'))
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
             if 'URL' in df.columns and pd.notna(df['URL']).any(): # Check if URL column has any valid data
                df['Platform'] = df['URL'].apply(infer_platform_from_url)
             else:
                df['Platform'] = "Unknown"
        # Otherwise, assume existing Platform column is good
    elif 'URL' in df.columns and pd.notna(df['URL']).any(): # Check if URL column has any valid data
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
        platforms_involved_str = ", ".join(p for p in platforms_involved if pd.notna(p) and p.strip() != "")

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
        if cluster_id == -1 or len(group) < 2: # -1 is noise, or too fewer members
            continue
        users = group['Influencer'].dropna().unique().tolist()
        for u1, u2 in combinations(users, 2):
            if G.has_edge(u1, u2):
                G[u1][u2]['weight'] += 1
            else:
                G.add_edge(u1, u2, weight=1)

    all_influencers = df['Influencer'].dropna().unique().tolist()
    # Capture primary platform for each influencer for node attributes
    influencer_platform_map = df.groupby('Influencer')['Platform'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').to_dict()

    for inf in all_influencers:
        if inf not in G.nodes():
            G.add_node(inf)
        G.nodes[inf]['platform'] = influencer_platform_map.get(inf, 'Unknown')


    pos = nx.spring_layout(G, seed=42, k=0.1, iterations=50)

    cluster_map = df.set_index('Influencer')['cluster'].to_dict()
    final_cluster_map = {node: cluster_map.get(node, 0) for node in G.nodes()} # Default to 0 if not in a cluster

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
def cached_network_graph(_df_for_graph):
    """
    Builds a user interaction network graph using the integrated function.
    Takes a potentially filtered DataFrame for the graph.
    """
    return build_user_interaction_graph(_df_for_graph)

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
            st.sidebar.warning("You uploaded multiple files. For best results, consider harmonizing column names (e.g., 'text', 'influencer') across files before upload if they differ significantly.")
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

    st.markdown("### 🔬 Preprocessed Data Sample")
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
                    st.markdown("### 🔝 Top Coordinated Narratives")
                    st.markdown("""
                        **Top Coordinated Narratives**: Identifies messages that appear across multiple original posts, indicating potential coordinated campaigns. 
                        The bar color reflects the amplification level via reposts, retweets, or quote tweets.
                    """)
                    narrative_summary = sim_df.groupby('shared_narrative').agg(
                        share_count=('similarity', 'count'),
                        influencers_involved=('account_id1', lambda x: ", ".join(x.astype(str).unique()[:5]) + ("..." if len(x.unique()) > 5 else "")),
                        platforms_involved=('platform1', lambda x: ", ".join(
                            sorted(list(set([p.strip() for p in x.astype(str).tolist() if p.strip() != ""])))
                        )),
                        repost_count=('shared_narrative', lambda x: get_repost_count(x.iloc[0]))
                    ).sort_values(by='share_count', ascending=False).reset_index()

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

                    st.dataframe(narrative_summary)

                    st.markdown("### 🔄 Full Similarity Pairs")
                    st.markdown("""
                        **Full Similarity Pairs**: Shows all pairs of posts with highly similar content, including source, timestamps, and URLs. 
                        This helps analysts verify context and trace narrative spread.
                    """)
                    display_sim_df = sim_df[[
                        'text1', 'account_id1', 'platform1', 'timestamp_share1', 'url1',
                        'text2', 'account_id2', 'platform2', 'timestamp_share2', 'url2',
                        'similarity'
                    ]].copy()

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

                    for col in ['Time 1', 'Time 2']:
                        if col in display_sim_df.columns:
                            display_sim_df[col] = pd.to_datetime(
                                display_sim_df[col], unit='s', utc=True, errors='coerce'
                            ).dt.strftime('%Y-%m-%d %H:%M')

                    st.data_editor(
                        display_sim_df,
                        column_config={
                            "URL 1": st.column_config.LinkColumn("URL 1"),
                            "URL 2": st.column_config.LinkColumn("URL 2")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Failed to generate narrative summary: {e}")
                    st.write("Debug: Check column names in `sim_df`:")
                    st.write(list(sim_df.columns))
            else:
                st.info("No significant similarities found above threshold between original posts.")
    else:
        st.subheader("🔗 Coordination by Shared URLs")
        st.markdown("""
            **Shared URLs Analysis**: Identifies content amplification through URL sharing across accounts. 
            This helps detect coordinated linking behavior, even when text is not identical.
        """)
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
            else:
                st.info("No URLs shared by multiple influencers.")
        else:
            st.info("No valid URLs to analyze.")
# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")
    st.markdown("---")

    # ✅ Debug: Show what we're working with
    st.write("🔍 **Available columns in filtered_df_global:**", list(filtered_df_global.columns))

    risk_df = filtered_df_global.copy()

    # 🔁 Force 'text' column from possible sources
    possible_text_columns = ['Hit Sentence', 'title', 'message', 'content', 'object_id', 'Body', 'FullText', 'opening text']
    found = False
    for col in possible_text_columns:
        if col in risk_df.columns:
            risk_df['text'] = risk_df[col].astype(str).fillna('').replace('nan', '').str.strip()
            st.info(f"✅ Created 'text' column from `{col}`")
            found = True
            break

    if not found:
        st.error("❌ No possible text column found. Available: " + str(list(risk_df.columns)))
        st.stop()

    # Clean and filter
    risk_df = risk_df[risk_df['text'] != ""].reset_index(drop=True)
    if risk_df.empty:
        st.error("❌ No valid text after cleaning.")
        st.stop()

    # 🔁 Convert UNIX timestamp to UTC datetime
    if 'timestamp_share' in risk_df.columns:
        risk_df['Timestamp'] = pd.to_datetime(risk_df['timestamp_share'], unit='s', utc=True, errors='coerce')
        risk_df = risk_df.dropna(subset=['Timestamp']).reset_index(drop=True)
    else:
        st.warning("⚠️ 'timestamp_share' not found. Using current time.")
        risk_df['Timestamp'] = pd.Timestamp.now(tz='UTC')

    # Ensure 'original_text' exists
    if 'original_text' not in risk_df.columns:
        risk_df['original_text'] = risk_df['text'].apply(extract_original_text)

    # ----------------------------
    # 🤖 Detected Coordination Clusters
    # ----------------------------
    st.markdown("### 🤖 Detected Coordination Clusters")
    st.markdown("""
        **Detected Coordination Clusters**: Groups posts into clusters based on content similarity, revealing coordinated campaigns. 
        Each cluster may represent a distinct narrative or disinformation effort.
    """)
    try:
        clustered_df = cached_clustering(risk_df)
        if 'cluster' not in clustered_df.columns:
            raise ValueError("Clustering did not return 'cluster' column")

        cluster_counts = clustered_df['cluster'].value_counts()
        if cluster_counts.empty:
            st.info("No clusters detected (e.g., all noise or only one post).")
        else:
            fig_clust = px.bar(
                cluster_counts,
                title="Cluster Sizes",
                labels={'value': 'Member Count', 'index': 'Cluster ID'},
                color=cluster_counts.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set3  # ✅ Correct: list of colors
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
    # 🕸️ User Interaction Network (Limited & Stable)
    # ----------------------------
    st.markdown("### 🕸️ User Interaction Network")
    st.markdown("""
        **User Interaction Network**: Visualizes connections between the most active influencers who share similar content. 
        Only the top influencers are shown to ensure performance and clarity. Use the slider to adjust the number of nodes.
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
            if 'original_text' not in graph_subset.columns:
                graph_subset['original_text'] = graph_subset['text'].apply(extract_original_text)

            # Build graph
            G, pos, cluster_map = cached_network_graph(graph_subset)

            if G is None or len(G.nodes()) == 0:
                st.info("No nodes to display in the network graph.")
            else:
                # Create edge traces
                edge_trace = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.append(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'
                    ))

                # Create node trace with safe colors
                node_clusters = [cluster_map.get(node, 0) for node in G.nodes()]
                unique_clusters = sorted(list(set(node_clusters)))
                
                # Cycle through Set3 colors if more clusters than colors
                from itertools import cycle
                color_cycle = cycle(px.colors.qualitative.Set3)
                color_map = {cluster: next(color_cycle) for cluster in unique_clusters}
                node_colors = [color_map[cluster] for cluster in node_clusters]

                node_trace = go.Scatter(
                    x=[pos[node][0] for node in G.nodes()],
                    y=[pos[node][1] for node in G.nodes()],
                    text=list(G.nodes()),
                    mode='markers+text',
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=node_colors,  # ✅ List of hex colors
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
    st.markdown("""
        **High-Risk Influencers**: Highlights accounts involved in 3 or more coordinated messages, indicating potential amplification roles. 
        These influencers may be central to spreading specific narratives across platforms.
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
