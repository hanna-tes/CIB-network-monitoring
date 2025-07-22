import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from modules.embedding_utils import compute_text_similarity

from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def cluster_texts(texts, threshold=0.75):
    """
    Cluster texts based on semantic similarity.
    """
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    return labels

def compute_pairwise_similarities(texts):
    """
    Compute pairwise cosine similarity between texts.
    """
    model = load_embedding_model()
    embeddings = model.encode(texts)
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix

def build_user_interaction_graph(df):
    """
    Build a user-post interaction graph where edges represent repeated user activity.
    """
    G = nx.Graph()
    if 'Source' in df.columns:
        users = df['Source'].dropna().unique()
        for user in users:
            G.add_node(user, type='user')
        
        for _, row in df.iterrows():
            user = row.get('Source')
            post_id = row.name
            if pd.notna(user):
                post_node = f"post_{post_id}"
                G.add_node(post_node, type='post', text=row.get('text', ''))
                G.add_edge(user, post_node)
    
    return G

def detect_coordinated_sharing(df, time_window_minutes=5):
    """
    Detect coordinated sharing based on identical URLs posted within a short time window.
    """
    if 'Timestamp' not in df.columns or 'URL' not in df.columns:
        return pd.DataFrame()

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp', 'URL'])
    df['URL'] = df['URL'].astype(str)

    coordinated_posts = []

    for url, group in df.groupby('URL'):
        sorted_group = group.sort_values('Timestamp')
        for i in range(len(sorted_group) - 1):
            t1 = sorted_group.iloc[i]['Timestamp']
            t2 = sorted_group.iloc[i+1]['Timestamp']
            delta = (t2 - t1).total_seconds() / 60
            if delta <= time_window_minutes:
                coordinated_posts.append(sorted_group.iloc[i])
                coordinated_posts.append(sorted_group.iloc[i+1])

    if coordinated_posts:
        return pd.DataFrame(coordinated_posts).drop_duplicates()
    return pd.DataFrame()

def detect_hashtag_overlap(df, min_shared=2):
    """
    Detect groups of users who share multiple hashtags.
    """
    if 'Source' not in df.columns or 'hashtags' not in df.columns:
        return []

    user_hashtags = df.dropna(subset=['hashtags']).groupby('Source')['hashtags'].apply(lambda x: ','.join(x)).to_dict()
    overlap_pairs = []

    users = list(user_hashtags.keys())
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            u1, u2 = users[i], users[j]
            h1 = set(re.findall(r'#\w+', user_hashtags[u1]))
            h2 = set(re.findall(r'#\w+', user_hashtags[u2]))
            shared = h1.intersection(h2)
            if len(shared) >= min_shared:
                overlap_pairs.append((u1, u2, list(shared)))

    return overlap_pairs
