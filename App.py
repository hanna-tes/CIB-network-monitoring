import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import networkx as nx
from io import StringIO
from datetime import datetime, timedelta
from collections import Counter
import re
import base64
import tempfile

st.set_page_config(page_title="Coordinated Sharing Detector", layout="wide")
st.title("🔍 Coordinated Sharing Detector")

# Sidebar
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
default_dataset = st.sidebar.checkbox("Use default dataset")

# Helper functions
def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

def extract_mentions(text):
    return re.findall(r"@(\w+)", text)

def extract_urls(text):
    return re.findall(r"https?://\S+", text)

# Load and process data
if uploaded_file or default_dataset:
    try:
        if default_dataset:
            df = pd.read_csv("data/example_social_data.csv")
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.lower()
        if 'timestamp' not in df.columns:
            st.error("Dataset must contain a 'timestamp' column.")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour

            if 'text' in df.columns:
                df['hashtags'] = df['text'].apply(lambda x: extract_hashtags(str(x)))
                df['mentions'] = df['text'].apply(lambda x: extract_mentions(str(x)))
                df['urls'] = df['text'].apply(lambda x: extract_urls(str(x)))
            else:
                st.warning("'text' column not found. Hashtags, mentions, and URLs won't be extracted.")

            # Summary Stats
            st.subheader("📊 Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Posts", len(df))
            with col2:
                st.metric("Unique Users", df['user_id'].nunique() if 'user_id' in df.columns else "N/A")
            with col3:
                st.metric("Time Range", f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

            # Top entities
            st.markdown("---")
            st.subheader("🔗 Top Shared Entities")
            if 'user_id' in df.columns:
                st.write("**Top Users:**")
                st.dataframe(df['user_id'].value_counts().head(10))
            if 'hashtags' in df.columns:
                all_hashtags = sum(df['hashtags'], [])
                st.write("**Top Hashtags:**")
                st.dataframe(pd.Series(all_hashtags).value_counts().head(10))
            if 'urls' in df.columns:
                all_urls = sum(df['urls'], [])
                st.write("**Top URLs:**")
                st.dataframe(pd.Series(all_urls).value_counts().head(10))

            # Temporal visualization
            st.markdown("---")
            st.subheader("🗓 Temporal Posting Patterns")
            daily_counts = df.groupby('date').size().reset_index(name='counts')
            chart = alt.Chart(daily_counts).mark_line(point=True).encode(
                x='date:T',
                y='counts:Q',
                tooltip=['date:T', 'counts']
            ).properties(width=700, height=300)
            st.altair_chart(chart, use_container_width=True)

            # Coordination detection
            st.markdown("---")
            st.subheader("🧪 Coordination Heuristics")
            if 'urls' in df.columns:
                window_sec = st.slider("Time window for coordination (seconds):", 1, 300, 30)
                suspicious = []
                grouped = df.explode('urls').dropna(subset=['urls'])
                for url in grouped['urls'].unique():
                    subset = grouped[grouped['urls'] == url].sort_values('timestamp')
                    times = subset['timestamp'].values
                    for i in range(len(times) - 1):
                        delta = (times[i+1] - times[i]) / np.timedelta64(1, 's')
                        if delta <= window_sec:
                            suspicious.append((url, subset.iloc[i]['user_id'], subset.iloc[i+1]['user_id'], times[i], times[i+1]))
                if suspicious:
                    st.write("**Detected coordinated URL sharing:**")
                    coord_df = pd.DataFrame(suspicious, columns=['URL', 'User A', 'User B', 'Time A', 'Time B'])
                    st.dataframe(coord_df)
                else:
                    st.success("No coordinated sharing detected within selected window.")

            # Network graph
            st.markdown("---")
            st.subheader("🖥 Network Graph (User Mentions)")
            if 'mentions' in df.columns and 'user_id' in df.columns:
                G = nx.DiGraph()
                for _, row in df.iterrows():
                    user = row['user_id']
                    for mention in row['mentions']:
                        G.add_edge(user, mention)
                if G.number_of_edges() > 0:
                    st.write(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                    pos = nx.spring_layout(G, k=0.15, iterations=20)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', ax=ax, font_size=8)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough mention data to create graph.")

            # Export tools
            st.markdown("---")
            st.subheader("📥 Export Findings")
            if 'coord_df' in locals():
                csv = coord_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="coordination_findings.csv">Download Coordinated Sharing CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to process file: {e}")
else:
    st.info("Please upload a dataset or use the default dataset to begin analysis.")
