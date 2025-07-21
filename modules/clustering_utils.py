import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from modules.embedding_utils import compute_text_similarity

# Placeholder clustering functions

def cluster_texts(texts, threshold=0.75):
    # Basic label assignment by grouping similar texts using pairwise similarity
    from sklearn.cluster import AgglomerativeClustering
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1-threshold, affinity='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)
    return labels

def build_user_interaction_graph(df):
    import networkx as nx
    G = nx.Graph()
    if 'Source' in df.columns:
        users = df['Source'].dropna().unique()
        for user in users:
            G.add_node(user)
        source_groups = df.groupby('Source')
        for source, group in source_groups:
            if len(group) > 1:
                posts = group.index.tolist()
                for i in range(len(posts)):
                    for j in range(i+1, len(posts)):
                        G.add_edge(source, f"post_{posts[i]}_{posts[j]}")
    return G

