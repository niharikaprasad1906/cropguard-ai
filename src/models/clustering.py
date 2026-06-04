
from sklearn.cluster import KMeans

def run_kmeans(X, k):
    return KMeans(n_clusters=k).fit_predict(X)
