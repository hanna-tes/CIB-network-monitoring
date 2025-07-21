from sklearn.cluster import DBSCAN
import numpy as np

def cluster_embeddings(embeddings, eps=0.3, min_samples=2):
    if len(embeddings) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    return clustering.labels_
