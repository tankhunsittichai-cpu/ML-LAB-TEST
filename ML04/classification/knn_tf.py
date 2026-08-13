from sklearn.neighbors import KNeighborsClassifier


def build_model(k=5):
    model = KNeighborsClassifier(
        n_neighbors=k
    )
    return model