from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import pandas as pd


def cluster_texts(df, eps=0.3, min_samples=2):

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    tfidf_matrix = vectorizer.fit_transform(df['original_text'])

    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(tfidf_matrix)

    df = df.copy()

    df['cluster'] = clustering.labels_

    return df



def build_user_interaction_graph(df):

    G = nx.Graph()

    # Simple: connect users who share same narrative (via high similarity or same cluster)

    grouped = df.groupby('cluster')

    for cluster_id, group in grouped:

        if cluster_id == -1 or len(group) < 2:

            continue

        users = group['Source'].tolist()

        for u1, u2 in combinations(users, 2):

            if G.has_edge(u1, u2):

                G[u1][u2]['weight'] += 1

            else:

                G.add_edge(u1, u2, weight=1)
    pos = nx.spring_layout(G, seed=42)

    cluster_map = dict(zip(df['Source'], df['cluster']))

    return G, pos, cluster_map




