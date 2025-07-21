import streamlit as st
import pandas as pd
import numpy as np
import re
import tempfile
import networkx as nx
from datetime import timedelta
from pyvis.network import Network  # For interactive graphs
import streamlit.components.v1 as components  # To display pyvis

# Custom modules
from modules.embedding_utils import compute_text_similarity
from modules.clustering_utils import cluster_texts, build_user_interaction_graph

# Streamlit setup
st.set_page_config(page_title="CIB Monitoring Dashboard", layout="wide")
st.title("🕵️ CIB Monitoring and Analysis Dashboard")

# Sidebar UI
st.sidebar.header("📂 Upload Social Media Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple CSV/Excel files",
    type=['csv', 'xlsx'],
    accept_multiple_files=True
)
st.sidebar.markdown("---")
similarity_threshold = st.sidebar.slider("🔍 Similarity Threshold", 0.0, 1.0, 0.75, 0.01)
exact_match_toggle = st.sidebar.checkbox("✅ Exact Text Match Only", value=False)

# --- Utility Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

# --- Standardize DataFrame Function ---
def standardize_dataframe(df, platform_name=None, source_name=None):
    col_map = {
        'Influencer': 'Source',
        'Opening Text': 'text',
        'Hit Sentence': 'text',
        'HitSentence': 'text',
        'authorMeta/name': 'Source',
        'createTimeISO': 'Timestamp',
        'webVideoUrl': 'URL',
        'Date': 'Timestamp',
        'date': 'Timestamp',
        'url': 'URL',
        'message': 'text',
        'title': 'text',
        'media_name': 'Source'
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

# --- Load & Process Data Once ---
combined_df = pd.DataFrame()
all_dfs = []

# Default dataset URL
default_url = "https://raw.githubusercontent.com/hanna-tes/CIB-network-monitoring/refs/heads/main/Togo_OR_Lome%CC%81_OR_togolais_OR_togolaise_AND_manifest%20-%20Jul%207%2C%202025%20-%205%2012%2053%20PM.csv "
default_url = default_url.strip()

if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-16', sep='\t', on_bad_lines='skip', low_memory=False)
            else:
                df = pd.read_excel(file)

            platform = 'TikTok' if ('authorMeta/name' in df.columns or 'authorMeta' in df.columns) else 'X'
            df_std = standardize_dataframe(df, platform_name=platform, source_name=file.name)
            all_dfs.append(df_std)

        except Exception as e:
            st.warning(f"Failed to read {file.name}: {e}")

else:
    try:
        df = pd.read_csv(default_url, encoding='utf-16', sep='\t', on_bad_lines='skip', low_memory=False)
        df_std = standardize_dataframe(df, platform_name='X', source_name='default_dataset')
        all_dfs.append(df_std)
    except Exception as e:
        st.warning(f"Failed to load default dataset: {e}")

# Concatenate and clean
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate
    combined_df.drop_duplicates(inplace=True)

    # Normalize Source
    combined_df['Source'] = combined_df['Source'].astype(str).str.strip().str.lower()
    invalid_sources = ['nan', '<nan>', 'none', 'unknown', '', 'null']
    combined_df.loc[combined_df['Source'].isin(invalid_sources), 'Source'] = 'unknown_user'
    combined_df['Source'] = combined_df['Source'].str.capitalize()

    # Ensure column order
    combined_df = combined_df[['Source', 'URL', 'Timestamp', 'text', 'Platform']].copy()
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce')

    # Clean text
    combined_df['text'] = combined_df['text'].astype(str).apply(clean_text)

    # === 🔍 Compute Derived Features ===

    # 1. Clustering
    try:
        cluster_labels = cluster_texts(combined_df['text'].tolist())
        combined_df['cluster'] = cluster_labels
    except Exception as e:
        st.warning(f"Clustering failed: {e}")
        combined_df['cluster'] = -1

    # 2. Text Similarity Pairs
    try:
        similar_pairs = compute_text_similarity(
            combined_df['text'].tolist(),
            threshold=similarity_threshold,
            exact=exact_match_toggle
        )
        st.session_state['similar_pairs'] = similar_pairs
    except Exception as e:
        st.warning(f"Similarity computation failed: {e}")
        st.session_state['similar_pairs'] = []

    # Cache final dataframe
    st.session_state['combined_df'] = combined_df

    st.success("✅ Data loaded, cleaned, and analyzed!")
    st.dataframe(combined_df.head(10))

    # Add download button
    csv = combined_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Data (CSV)",
        data=csv,
        file_name="cib_cleaned_data.csv",
        mime="text/csv"
    )

else:
    st.info("📤 Please upload files or connect to the default dataset.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Text Similarity", "📎 About"])

# ========================
# TAB 1: DASHBOARD
# ========================
with tab1:
    st.subheader("1. Data Overview")
    st.dataframe(combined_df)

    st.subheader("2. Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", len(combined_df))
    col2.metric("Unique Sources", combined_df['Source'].nunique())
    col3.metric("Platforms", combined_df['Platform'].nunique())

    # --- 📈 Posting Timeline ---
    st.subheader("3. Posting Activity Over Time")
    timeline = combined_df.dropna(subset=['Timestamp']).copy()
    timeline['hour'] = timeline['Timestamp'].dt.floor('h')
    post_counts = timeline.groupby('hour').size().reset_index(name='counts')
    chart = alt.Chart(post_counts).mark_line(point=True).encode(
        x=alt.X('hour:T', title='Time'),
        y=alt.Y('counts:Q', title='Number of Posts'),
        tooltip=['hour', 'counts']
    ).properties(title='Posting Frequency', width=700)
    st.altair_chart(chart)

    # --- 👥 Most Active Sources ---
    st.subheader("4. Top 10 Most Active Sources")
    top_users = combined_df['Source'].value_counts().head(10)
    st.bar_chart(top_users)

    # --- ⚠️ High-Risk Accounts Detection ---
    st.subheader("5. 🔴 High-Risk Accounts (Coordination Score)")

    def compute_risk_scores(df, similar_pairs):
        scores = pd.Series(0, index=df.index, dtype=float)
        user_risk = df['Source'].copy().to_frame()
        user_risk['risk_score'] = 0.0
        user_risk['flags'] = ""

        url_shares = df.dropna(subset=['URL'])
        suspicious_urls = []
        for url, group in url_shares.groupby('URL'):
            times = group['Timestamp'].sort_values()
            deltas = times.diff().dropna()
            rapid = (deltas <= timedelta(minutes=5)).sum()
            if rapid >= 2:
                suspicious_urls.append(url)

        fast_reposters = url_shares[url_shares['URL'].isin(suspicious_urls)]['Source'].value_counts()

        # Get similar posters
        similar_posters = set()
        for i, j, _ in similar_pairs:
            similar_posters.add(df.iloc[i]['Source'])
            similar_posters.add(df.iloc[j]['Source'])

        # Aggregate risk per user
        user_scores = {}
        user_flags = {}

        for src in df['Source'].unique():
            score = 0
            flags = []

            # Flag 1: Fast URL Reposter
            count = fast_reposters.get(src, 0)
            if count > 0:
                score += count * 0.3
                flags.append(f"Fast URL share ({count})")

            # Flag 2: In Similar Text Group
            if src in similar_posters:
                sim_count = sum(1 for p in similar_pairs if df.iloc[p[0]]['Source'] == src or df.iloc[p[1]]['Source'] == src)
                score += sim_count * 0.2
                flags.append(f"Similar text ({sim_count})")

            # Flag 3: High Cluster Participation
            user_clusters = df[df['Source'] == src]['cluster']
            if (user_clusters != -1).any():
                common_cluster_size = df['cluster'].value_counts().get(user_clusters.mode().iloc[0], 1)
                if common_cluster_size > 3:
                    score += min(common_cluster_size / 10, 0.5)  # cap at 0.5

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
            high_risk.style.background_gradient(cmap='Reds', subset=['Risk Score']),
            use_container_width=True
        )
    else:
        st.info("No high-risk accounts detected.")

    # --- 🌐 Interactive Network Graph (pyvis) ---
    st.subheader("6. Interactive User Interaction Network")
    G = build_user_interaction_graph(combined_df)

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
    net.from_nx(G)

    # Color nodes by risk score
    risk_dict = risk_df.set_index('Source')['Risk Score'].to_dict()
    for node in net.nodes:
        src = node['label']
        score = risk_dict.get(src, 0)
        hue = int(120 * (1 - min(score / 1.0, 1)))  # Green (120) → Red (0)
        node['color'] = f"hsl({hue}, 80%, 60%)"
        node['title'] = f"<b>{src}</b><br>Risk Score: {score:.2f}"

    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "repulsion": { "nodeDistance": 120 }
      },
      "interaction": { "tooltip": true }
    }
    """)

    # Save and render
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as fp:
        net.save_graph(fp.name)
        with open(fp.name, 'r', encoding='utf-8') as f:
            graph_html = f.read()

    components.html(graph_html, height=650)

    # --- 🧩 Text Clusters ---
    st.subheader("7. Text Clusters Distribution")
    fig = px.histogram(combined_df, x='cluster', title="Text Clusters", nbins=20)
    st.plotly_chart(fig)

# ========================
# TAB 2: TEXT SIMILARITY
# ========================
with tab2:
    st.subheader("🔍 Detected Similar Text Pairs")
    similar_pairs = st.session_state['similar_pairs']

    if similar_pairs:
        st.markdown(f"Found **{len(similar_pairs)}** similar pairs above threshold `{similarity_threshold}`:")
        for idx1, idx2, score in sorted(similar_pairs, key=lambda x: -x[2]):
            with st.expander(f"🔁 Match • Score: {score:.3f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Post 1", combined_df.iloc[idx1]['text'], height=120)
                    st.caption(f"Source: {combined_df.iloc[idx1]['Source']} | Platform: {combined_df.iloc[idx1]['Platform']}")
                with col2:
                    st.text_area("Post 2", combined_df.iloc[idx2]['text'], height=120)
                    st.caption(f"Source: {combined_df.iloc[idx2]['Source']} | Platform: {combined_df.iloc[idx2]['Platform']}")
    else:
        st.info("No similar text found at current settings.")

# ========================
# TAB 3: ABOUT
# ========================
with tab3:
    st.markdown("""
    ### 🔍 CIB Monitoring Dashboard

    Detects signs of **Coordinated Inauthentic Behavior (CIB)** across platforms.

    **Features:**
    - 📥 Multi-format upload (CSV/XLSX)
    - 🔀 Schema auto-standardization
    - 🔁 Fast repost detection
    - 🧠 Semantic similarity & clustering
    - 👥 Interactive network graph
    - 🔴 High-risk account scoring
    - 💾 Export cleaned data

    Built with:  
    `Streamlit`, `PyVis`, `Hugging Face`, `NetworkX`, `Altair`

    💡 Tip: Use low thresholds (~0.5–0.6) for broad matches; increase for precision.
    """)
