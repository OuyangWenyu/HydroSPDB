"""According to some attributes in GAGES-II dataset, cluster points to different catogries, then train different
combinations """
from sklearn.cluster import KMeans


def cluster_attr_train(X, num_cluster):
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
    return kmeans, kmeans.labels_


def cluster_attr_test(kmeans, x):
    labels = kmeans.predict(x)
    return labels
