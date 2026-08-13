from sklearn.metrics import silhouette_score

def evaluate_cluster(X, labels):

    score = silhouette_score(
        X,
        labels
    )

    return score
