from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def evaluate_model(y_true, y_pred):

    acc = accuracy_score(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred
    )

    return acc, report