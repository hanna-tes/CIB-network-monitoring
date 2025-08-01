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
    cleaned = re.sub(r'^(RT|rt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    return cleaned

@st.cache_data(show_spinner="📥 Loading default dataset...")
def load_default_dataset():
    url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/TogoJULYData%20-%20Sheet1.csv"
    try:
        df = pd.read_csv(url)
        st.sidebar.success("✅ Default data loaded")
        return df
    except Exception as e:
        st.error(f"Failed to load default dataset: {e}")
        return pd.DataFrame()

# --- Preprocessing Function ---
def preprocess_data(df):
    """Preprocesses the DataFrame with robust column mapping."""
    if df.empty:
        return df

    df = df.drop_duplicates().reset_index(drop=True)

    # --- EARLY COLUMN MAPPING (Normalized) ---
    col_map = {
        # 💬 Text Content
        'Hit Sentence': 'text', 'Headline': 'text', 'opening text': 'text',
        'Opening Text': 'text', 'message': 'text', 'content': 'text',
        'description': 'text', 'Body': 'text', 'FullText': 'text',
        'title': 'text', 'Title': 'text',  # Media data uses 'title'
        'Caption': 'text', 'caption': 'text',

        # 👤 Influencer / Author
        'Influencer': 'Influencer', 'author': 'Influencer', 'username': 'Influencer',
        'user': 'Influencer', 'authorMeta/name': 'Influencer', 'creator': 'Influencer',
        'authorname': 'Influencer', 'media_name': 'Influencer', 'channeltitle': 'Influencer',
        'source': 'Influencer', 'Source': 'Influencer',

        # 📅 Timestamps
        'Date': 'Timestamp', 'createTimeISO': 'Timestamp', 'published_date': 'Timestamp',
        'pubDate': 'Timestamp', 'created_at': 'Timestamp', 'Alternate Date Format': 'Timestamp',
        'publish_date': 'Timestamp', 'time': 'Timestamp',
    }

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
    required_cols = ["Influencer", "text"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns after mapping: {missing_cols}")

        suggestions = {
            'Influencer': ['influencer', 'author', 'user', 'username', 'source', 'media_name'],
            'text': ['hit sentence', 'opening text', 'headline', 'message', 'content', 'title']
        }

        for col in missing_cols:
            close_matches = [
                c for c in df.columns
                if any(sugg in c.lower().replace(" ", "") for sugg in suggestions.get(col, []))
            ]
            if close_matches:
                st.info(f"💡 Did you mean to map `{close_matches[0]}` → `{col}`?")
            else:
                if col == "Influencer":
                    df['Influencer'] = "Unknown_User"
                    st.warning("⚠️ Using 'Unknown_User' for Influencer.")
                elif col == "text":
                    st.error("🚫 No text column found. Cannot proceed.")
                    st.stop()

        for col in required_cols:
            if col not in df.columns:
                st.error(f"🛑 Still missing: '{col}' → Cannot continue.")
                st.stop()

    # --- Ensure 'text' is string and clean ---
    df['text'] = df['text'].astype(str)
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != ""]
    df = df[df['text'].str.lower() != "nan"].reset_index(drop=True)

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'^QT.*?;.*', lambda m: m.group(0).split(';')[0], text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r"rt @\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"qt @\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0].reset_index(drop=True)

    # --- Use 'text' as 'content_id' (for similarity analysis) ---
    df['content_id'] = df['text'].str[:50] + "..."  # Shortened unique identifier

    # --- Extract Hashtags ---
    df['hashtags'] = df['text'].str.findall(r'#\w+') \
                       .apply(lambda tags: [tag.lower() for tag in tags if len(tag) > 1])

    # --- Timestamp Parsing ---
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = pd.Timestamp.now(tz='UTC')
        st.warning("⚠️ No timestamp column found. Using current time.")

    date_formats = [
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f', '%d %b %Y %H:%M:%S', '%A, %d %b %Y %H:%M:%S',
        '%b %d, %Y %I:%M%p', '%d %b %Y %I:%M%p', '%Y-%m-%d %H:%M:%S%z',
    ]

    def parse_timestamp(timestamp):
        if pd.isna(timestamp):
            return pd.NaT
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce')
                if pd.notna(parsed):
                    return parsed
            except (ValueError, TypeError):
                continue
        return pd.to_datetime(timestamp, infer_datetime_format=True, errors='coerce')

    df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)
    df = df.dropna(subset=['Timestamp']).reset_index(drop=True)

    # --- Create 'Platform' from URL ---
    url_cols = ['URL', 'url', 'webVideoUrl', 'link', 'post_url']
    url_found = False
    for col in url_cols:
        if col in df.columns:
            df['URL'] = df[col].astype(str).replace('nan', '').replace('None', '')
            url_found = True
            break

    if url_found:
        df['Platform'] = df['URL'].apply(infer_platform_from_url)
    else:
        df['Platform'] = "Unknown"
        st.sidebar.warning("⚠️ No URL column found → all platforms marked as 'Unknown'")

    # --- Extract original text (remove RT) ---
    df['original_text'] = df['text'].apply(extract_original_text)

    return df

# Vectorized similarity function
def find_textual_similarities(df, threshold=0.85):
    clean_df = df[['original_text', 'Influencer', 'Timestamp', 'URL', 'Platform']].copy()
    clean_df['original_text'] = clean_df['original_text'].astype(str)
    clean_df = clean_df.dropna(subset=['original_text', 'Influencer', 'Timestamp', 'Platform'])
    clean_df = clean_df[clean_df['original_text'].str.strip() != ""]

    texts = clean_df['original_text'].tolist()
    if len(texts) < 2:
        st.info("Not enough valid original posts for similarity analysis.")
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except:
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
            'platform1': row1['Platform'],
            'time1': row1['Timestamp'],
            'url1': row1['URL'],
            'text2': row2['original_text'],
            'influencer2': row2['Influencer'],
            'platform2': row2['Platform'],
            'time2': row2['Timestamp'],
            'url2': row2['URL'],
            'similarity': round(sim_matrix[i, j], 3),
            'shared_narrative': narrative_snippet,
            'platforms_involved': platforms_involved_str
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
    except:
        _df = _df.copy()
        _df['cluster'] = 0
        return _df

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df):
    try:
        from modules.clustering_utils import build_user_interaction_graph
        return build_user_interaction_graph(_df)
    except:
        G = nx.Graph()
        nodes = _df['Influencer'].dropna().unique()[:10]
        for u in nodes:
            G.add_node(u)
            for v in nodes:
                if u != v:
                    G.add_edge(u, v, weight=1)
        pos = nx.spring_layout(G, seed=42)
        cluster_map = {n: 0 for n in G.nodes}
        return G, pos, cluster_map

# --- Sidebar: Data Source ---
st.sidebar.header("📥 Data Source")
st.sidebar.info("Using default demo dataset")
df = load_default_dataset()

# Exit if no data
if df is None or df.empty:
    st.warning("No data available. Please upload a valid dataset.")
    st.stop()

# --- Preprocess ---
df = preprocess_data(df)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")
available_platforms = df['Platform'].dropna().astype(str).unique().tolist()
platforms = st.sidebar.multiselect(
    "Platforms",
    options=available_platforms,
    default=available_platforms
)

# Apply platform filter
filtered_df = df[df['Platform'].isin(platforms)].copy() if platforms else df.copy()

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

    # Top 10 Influencers
    top_influencers = filtered_df['Influencer'].value_counts().head(10)
    fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'Influencer'})
    st.plotly_chart(fig_src, use_container_width=True)
    st.caption("Shows the most active influencers based on number of posts.")

    # Top 10 Outlets
    if 'Outlet' in filtered_df.columns:
        top_outlets = filtered_df['Outlet'].value_counts().head(10)
        fig_out = px.bar(top_outlets, title="Top 10 Outlets", labels={'value': 'Articles', 'index': 'Outlet'})
        st.plotly_chart(fig_out, use_container_width=True)
        st.caption("Ranks media outlets by number of published articles or posts.")

    # Top 10 Channels
    if 'Channel' in filtered_df.columns:
        top_channels = filtered_df['Channel'].value_counts().head(10)
        fig_chan = px.bar(top_channels, title="Top 10 Channels", labels={'value': 'Posts', 'index': 'Channel'})
        st.plotly_chart(fig_chan, use_container_width=True)
        st.caption("Displays the most active social media or news channels.")

    # Top 10 Hashtags
    if 'hashtags' in filtered_df.columns:
        all_hashtags = [tag for tags in filtered_df['hashtags'] for tag in tags]
        if all_hashtags:
            hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
            fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags", labels={'value': 'Frequency', 'index': 'Hashtag'})
            st.plotly_chart(fig_ht, use_container_width=True)
            st.caption("Highlights the most frequently used hashtags across posts.")
        else:
            st.info("No valid hashtags found in the data.")

# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🧠 Narrative Detection & Coordination")
    MAX_ROWS = st.sidebar.slider("Max posts to analyze", 100, 1000, 300)
    analysis_df = filtered_df.head(MAX_ROWS).copy()

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
        st.caption("Identifies the most repeated messages across different accounts, indicating potential coordination.")

        st.dataframe(narrative_summary)
        st.markdown("### 🔄 Full Similarity Pairs")
        display_sim_df = sim_df.copy()
        display_sim_df['url1'] = display_sim_df['url1'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) and x.strip() != "" else "")
        display_sim_df['url2'] = display_sim_df['url2'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) and x.strip() != "" else "")
        st.markdown(display_sim_df.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.info("No significant similarities found above threshold.")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🚨 High-Risk Accounts & Networks")

    try:
        clustered_df = cached_clustering(filtered_df)
        if 'cluster' not in clustered_df.columns:
            raise ValueError("Clustering did not return 'cluster' column")
        cluster_counts = clustered_df['cluster'].value_counts()
        st.markdown("### 🤖 Detected Coordination Clusters")
        fig_clust = px.bar(
            cluster_counts,
            title="Cluster Sizes",
            labels={'value': 'Member Count', 'index': 'Cluster ID'},
            color=cluster_counts.index.astype(str),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_clust, use_container_width=True)
        st.caption("Groups posts into clusters based on content similarity, highlighting coordinated campaigns.")
        st.dataframe(clustered_df[['Influencer', 'text', 'Timestamp', 'cluster']])
    except Exception as e:
        st.warning(f"⚠️ Clustering failed: {e}")

    st.markdown("### 🕸️ User Interaction Network")
    try:
        G, pos, cluster_map = cached_network_graph(clustered_df if 'clustered_df' in locals() else filtered_df)
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'))
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=list(G.nodes()),
            mode='markers+text',
            textposition="top center",
            marker=dict(
                size=12,
                color=[cluster_map.get(node, 0) for node in G.nodes()],
                colorscale='Set3',
                colorbar=dict(title="Clusters"),
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
        st.caption("Visualizes connections between influencers based on shared narratives.")
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
                st.caption("Highlights influencers who repeatedly share similar messages, indicating potential coordination.")
            else:
                st.info("No influencer shared 3+ narratives.")
        else:
            st.info("No coordinated narratives detected.")
    except Exception as e:
        st.warning(f"Risk analysis failed: {e}")
