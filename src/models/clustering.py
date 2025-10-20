from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans

def kmeans_groups(embs: list[list[float]], k: int = 4) -> list[int]:
    X = np.array(embs)
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X)
    return labels.tolist()
