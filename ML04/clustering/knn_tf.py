from sklearn.neighbors import NearestNeighbors


def build_knn(n_neighbors=5):

    model = NearestNeighbors(
        n_neighbors=n_neighbors
    )

    return model