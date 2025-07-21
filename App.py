# App.py - CIB Monitoring Dashboard (Multi-Platform Support)

import streamlit as st
import pandas as pd
import numpy as np
import re
import tempfile
import networkx as nx
from datetime import timedelta
from pyvis.network import Network
import streamlit.components.v1 as components
import altair as alt  # Critical: Must be imported!

# Try to load custom modules with fallbacks
try:
    from modules.embedding_utils import compute_text_similarity
except ImportError:
    def compute_text_similarity(texts, threshold=0.75, exact=False):
        return []

try:
    from modules.clustering_utils import cluster_texts, build_user_interaction_graph
except ImportError:
    def cluster_texts(texts):
        return [-1] * len(texts)

    def build_user_interaction_graph(df):
        G = nx.Graph()
        sources = df[df['Source'] != 'Unknown_User']['Source'].dropna().unique()[:50]
        for src in sources:
            G.add_node(src)
        return G

# -------------------------------
# Utility Functions
# -------------------------------
@st.cache_data
def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def extract_hashtags(text):
    if pd.isna(text):
        return []
    return re.findall(r"#(\w+)", str(text))

# Fix QT @AccountName pattern in Hit Sentence (X/Meltwater only)
def fix_hit_sentence(row):
    opening_text = str(row.get('Opening Text', ''))
    hit_sentence = str(row.get('Hit Sentence', ''))

    if opening_text.startswith('QT') and '@' in opening_text:
        account_name_match = re.match(r'^QT (@[\w]+):', opening_text)
        if account_name_match:
            account_name = account_name_match.group(1)
            if hit_sentence.startswith(f"@{account_name}") and not hit_sentence.startswith('QT'):
                return f"QT {account_name}: {hit_sentence[2:]}"
            elif not hit_sentence.startswith(f"@{account_name}"):
                return f"QT {account_name}: {hit_sentence}"
    return hit_sentence

# -------------------------------
# Standardize DataFrame Function (Multi-Platform)
# -------------------------------
def standardize_dataframe(df, platform_name=None, source_name=None):
    col_map = {
        # X / Meltwater
        'Influencer': 'Source',
        'Opening Text': 'text',
        'Hit Sentence': 'text',
        'Date': 'Timestamp',
        'url': 'URL',

        # TikTok
        'authorMeta/name': 'Source',
        'webVideoUrl': 'URL',
        'createTimeISO': 'Timestamp',
        'text': 'text',

        # Facebook
        'post_url': 'URL',
        'created_time': 'Timestamp',
        'message': 'text',
        'author_name': 'Source',

        # Telegram
        'channeltitle': 'Source',
        'date': 'Timestamp',
        'message': 'text',
        'url': 'URL',

        # Media Articles
        'media_name': 'Source',
        'publish_date': 'Timestamp',
        'title': 'text',
        'url': 'URL'
    }

    rename_dict = {col: col_map[col] for col in df.columns if col in col_map}
    df_renamed = df.rename(columns=rename_dict)

    standardized = {}
    for col in ['Source', 'URL', 'Timestamp', 'text']:
        matched_cols = [orig for orig, std in col_map.items() if std == col and orig in df.columns]
        if matched_cols:
            combined = df[matched_cols].apply(
                lambda row: next((str(val) for val in row if pd.notna(val)), np.nan), axis=1
            )
            standardized[col] = combined
        else:
            standardized[col] = np.nan

    df_std = pd.DataFrame(standardized)[['Source', 'URL', 'Timestamp', 'text']].copy()
    df_std['Platform'] = platform_name or source_name or 'Unknown'

    return df_std

# -------------------------------
# Load Default Dataset (GitHub Raw URL)
# -------------------------------
@st.cache_data(ttl=600)
def load_default_dataset():
    default_url = (
        "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/ "
        "refs/heads/main/Togo_OR_Lome%CC%81_OR_togolais_OR_togolaise_AND_manifest%20-%20Jul%207%2C%202025%20-%205%2012%2053%20PM.csv"
    )
    try:
        df = pd.read_csv(default_url, encoding='utf-16', sep='\t', on_bad_lines='skip', low_memory=False)
        st.info("✅ Successfully loaded **default dataset** from GitHub.")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load default dataset: {e}")
        return None

# -------------------------------
# Main: Data Input Selection
# -------------------------------
st.set_page_config(page_title="CIB Monitoring Dashboard", layout="wide")
st.title("🕵️ CIB Monitoring and Analysis Dashboard")

st.sidebar.header("📂 Choose Data Source")
data_option = st.sidebar.radio(
    "Select Input Method",
    options=["📁 Upload Files", "🌐 Use Default Dataset"],
    help="Upload your own data or explore using the default dataset."
)

combined_df = pd.DataFrame()

if data_option == "📁 Upload Files":
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV or Excel files", type=['csv', 'xlsx'], accept_multiple_files=True
    )

    if uploaded_files:
        all_dfs = []
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file, encoding='utf-16', sep='\t', on_bad_lines='skip', low_memory=False)
                else:
                    df = pd.read_excel(file)

                # Detect platform based on columns
                if 'authorMeta/name' in df.columns:
                    platform = 'TikTok'
                elif 'author_name' in df.columns and 'post_url' in df.columns:
                    platform = 'Facebook'
                elif 'channeltitle' in df.columns:
                    platform = 'Telegram'
                elif 'media_name' in df.columns:
                    platform = 'Media'
                else:
                    platform = 'X'  # Default

                # Apply platform-specific fixes
                if platform == 'X' and 'Hit Sentence' in df.columns and 'Opening Text' in df.columns:
                    df['Hit Sentence'] = df.apply(fix_hit_sentence, axis=1)

                # Standardize
                df_std = standardize_dataframe(df, platform_name=platform, source_name=file.name)
                all_dfs.append(df_std)

            except Exception as e:
                st.warning(f"Failed to read {file.name}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            st.success(f"✅ Loaded and standardized {len(uploaded_files)} file(s) across platforms.")

elif data_option == "🌐 Use Default Dataset":
    with st.spinner("📥 Loading default dataset..."):
        df = load_default_dataset()
        if df is not None:
            # Assume it's X data (Meltwater-style)
            if 'Hit Sentence' in df.columns and 'Opening Text' in df.columns:
                df['Hit Sentence'] = df.apply(fix_hit_sentence, axis=1)
            combined_df = standardize_dataframe(df, platform_name='X', source_name='default_dataset')
            st.success("✅ Default dataset loaded and standardized!")

# -------------------------------
# Post-Processing (All Modes)
# -------------------------------
if not combined_df.empty:
    # Deduplicate
    combined_df.drop_duplicates(inplace=True)

    # Normalize Source
    combined_df['Source'] = combined_df['Source'].astype(str).str.strip().str.lower()
    invalid_sources = ['nan', '<nan>', 'none', 'unknown', '', 'null']
    combined_df.loc[combined_df['Source'].isin(invalid_sources), 'Source'] = 'unknown_user'
    combined_df['Source'] = combined_df['Source'].str.capitalize()

    # Ensure correct column order
    combined_df = combined_df[['Source', 'URL', 'Timestamp', 'text', 'Platform']].copy()

    # Parse Timestamp efficiently
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce', format='ISO8601')

    # Add cleaned text and hashtags
    combined_df['cleaned_text'] = combined_df['text'].apply(clean_text)
    combined_df['hashtags'] = combined_df['text'].apply(extract_hashtags)

    # Warn about missing URLs
    if combined_df['URL'].isna().all():
        st.warning("⚠️ All URLs are missing. Check column naming in your data.")

    # Clustering
    try:
        valid_texts = [t for t in combined_df['cleaned_text'] if t.strip()]
        if valid_texts:
            combined_df['cluster'] = cluster_texts(valid_texts)
        else:
            combined_df['cluster'] = -1
    except Exception as e:
        st.warning(f"⚠️ Clustering failed: {e}")
        combined_df['cluster'] = -1

    # Similarity
    try:
        similar_pairs = compute_text_similarity(
            [t for t in combined_df['cleaned_text']],
            threshold=0.75
        )
        st.session_state['similar_pairs'] = similar_pairs
    except:
        st.session_state['similar_pairs'] = []

    # Cache final result
    st.session_state['combined_df'] = combined_df

    # Show preview
    st.subheader("📊 Standardized Data Preview (5 Columns)")
    st.dataframe(combined_df.head(10))

    # Export button
    csv = combined_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv,
        file_name="cib_cleaned_data.csv",
        mime="text/csv"
    )

else:
    st.info("📤 Please select a data source to begin analysis.")
    st.stop()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🔍 Text Similarity", "ℹ️ About"])

# ========================
# TAB 1: DASHBOARD
# ========================
with tab1:
    st.subheader("1. Summary Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Posts", len(combined_df))
    c2.metric("Unique Sources", combined_df['Source'].nunique())
    c3.metric("Platforms", combined_df['Platform'].nunique())
    c4.metric("Non-null URLs", combined_df['URL'].notna().sum())

    # Platform distribution
    st.markdown("**Platform Distribution:**")
    platform_counts = combined_df['Platform'].value_counts()
    st.bar_chart(platform_counts)

    # --- Posting Timeline ---
    st.subheader("2. Posting Activity Over Time")
    timeline = combined_df.dropna(subset=['Timestamp']).copy()
    timeline['hour'] = timeline['Timestamp'].dt.floor('h')
    post_counts = timeline.groupby('hour').size().reset_index(name='counts')
    chart = alt.Chart(post_counts).mark_line(point=True).encode(
        x=alt.X('hour:T', title='Time'),
        y=alt.Y('counts:Q', title='Number of Posts'),
        tooltip=['hour:T', 'counts']
    ).properties(title="Posting Frequency")
    st.altair_chart(chart, use_container_width=True)

    # --- Trending Hashtags ---
    st.subheader("3. 📈 Top Trending Hashtags")
    all_tags = [tag.lower() for tags in combined_df['hashtags'] for tag in tags]
    if all_tags:
        freq = pd.Series(all_tags).value_counts().head(15)
        st.bar_chart(freq)
    else:
        st.info("No hashtags found in the data.")

    # --- High-Risk Accounts ---
    st.subheader("4. 🔴 High-Risk Accounts")
    def compute_risk_scores(df, similar_pairs):
        user_scores = {}
        user_flags = {}

        url_posts = df.dropna(subset=['URL']).sort_values('Timestamp')
        suspicious_urls = []
        for url, group in url_posts.groupby('URL'):
            deltas = group['Timestamp'].diff().dropna()
            rapid_shares = (deltas <= timedelta(minutes=5)).sum()
            if rapid_shares >= 2:
                suspicious_urls.append(url)
        fast_reposters = url_posts[url_posts['URL'].isin(suspicious_urls)]['Source'].value_counts()

        similar_posters = set()
        for i, j, _ in similar_pairs:
            similar_posters.add(df.iloc[i]['Source'])
            similar_posters.add(df.iloc[j]['Source'])

        for src in df['Source'].unique():
            score = 0.0
            flags = []

            count = fast_reposters.get(src, 0)
            if count > 0:
                score += count * 0.3
                flags.append(f"Fast URL ({count})")

            sim_count = sum(1 for p in similar_pairs if df.iloc[p[0]]['Source'] == src or df.iloc[p[1]]['Source'] == src)
            if src in similar_posters:
                score += sim_count * 0.2
                flags.append(f"Similar text ({sim_count})")

            cluster_label = df[df['Source'] == src]['cluster']
            if len(cluster_label) > 1 and cluster_label.mode().iloc[0] != -1:
                size = df['cluster'].value_counts().get(cluster_label.mode().iloc[0], 1)
                if size > 3:
                    score += min(size / 10, 0.5)

            user_scores[src] = round(score, 2)
            user_flags[src] = ", ".join(flags) if flags else "Low activity"

        risk_df = pd.DataFrame({
            'Source': list(user_scores.keys()),
            'Risk Score': list(user_scores.values()),
            'Flags': list(user_flags.values())
        }).sort_values(by='Risk Score', ascending=False)

        return risk_df

    risk_df = compute_risk_scores(combined_df, st.session_state['similar_pairs'])
    high_risk = risk_df[risk_df['Risk Score'] > 0]

    if not high_risk.empty:
        st.dataframe(
            high_risk.style.background_gradient(cmap='Reds', subset=["Risk Score"]),
            use_container_width=True
        )
    else:
        st.info("No high-risk accounts detected.")

    # --- Interactive Network ---
    st.subheader("5. 🌐 User Interaction Network")
    G = build_user_interaction_graph(combined_df)
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)

    risk_dict = risk_df.set_index('Source')['Risk Score'].to_dict()
    for node in net.nodes:
        src = node['label']
        score = risk_dict.get(src, 0)
        hue = int(120 * (1 - min(score, 1)))
        node['color'] = f"hsl({hue}, 80%, 60%)"
        node['title'] = f"<b>{src}</b><br>Risk Score: {score:.2f}"

    net.set_options("""
    var options = {
      "physics": {"enabled": true, "repulsion": {"nodeDistance": 120}},
      "interaction": {"hover": true}
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        net.save_graph(tmpfile.name)
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)

# ========================
# TAB 2: TEXT SIMILARITY
# ========================
with tab2:
    st.subheader("🔍 Detected Similar Text Pairs")
    pairs = st.session_state['similar_pairs']

    if pairs:
        for idx1, idx2, score in sorted(pairs, key=lambda x: -x[2]):
            with st.expander(f"🔁 Match • Score: {score:.3f}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Source:** {combined_df.iloc[idx1]['Source']} | **Platform:** {combined_df.iloc[idx1]['Platform']}")
                    st.text_area("Post 1", combined_df.iloc[idx1]['text'], height=120)
                with c2:
                    st.markdown(f"**Source:** {combined_df.iloc[idx2]['Source']} | **Platform:** {combined_df.iloc[idx2]['Platform']}")
                    st.text_area("Post 2", combined_df.iloc[idx2]['text'], height=120)
    else:
        st.info("No similar text pairs found at current threshold.")

# ========================
# TAB 3: ABOUT
# ========================
with tab3:
    st.markdown("""
    ### 🕵️ CIB Monitoring Dashboard

    Detects coordinated inauthentic behavior across:
    - **X (Twitter/Meltwater)**
    - **TikTok**
    - **Facebook**
    - **Telegram**
    - **Media Articles**

    **Features:**
    - 🔄 Auto-standardizes multi-source data
    - 📥 Upload or use default dataset
    - 🧠 Semantic similarity & clustering
    - 📈 Hashtag trends
    - 👥 Cross-platform network graph
    - 🔴 Risk scoring engine

    Built with: `Streamlit`, `PyVis`, `Altair`, `Pandas`

    💡 Tip: Start with the default dataset to explore!
    """)
