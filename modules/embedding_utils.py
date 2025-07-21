# modules/embedding_utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_text_similarity(texts, threshold=0.75, exact=False):
    similar_pairs = []

    if exact:
        seen = {}
        for i, text in enumerate(texts):
            if text in seen:
                similar_pairs.append((seen[text], i, 1.0))
            else:
                seen[text] = i
        return similar_pairs

    # TF-IDF Vectorization for text similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = similarity_matrix[i, j]
            if sim >= threshold:
                similar_pairs.append((i, j, sim))

    return similar_pairs
