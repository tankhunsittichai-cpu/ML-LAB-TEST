import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_loader import load_data

# -----------------
# PATH
# -----------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "age_gender(1).csv"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "output_folder"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

print("CSV PATH =", CSV_PATH)
print("FOUND =", os.path.exists(CSV_PATH))

# -----------------
# LOAD DATA
# -----------------

X, df = load_data(CSV_PATH)

print("Load Success")

# -----------------
# KMEANS
# -----------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X)

df["cluster"] = labels

# -----------------
# EVALUATION
# -----------------

score = silhouette_score(
    X,
    labels
)

print("Silhouette Score =", score)

# -----------------
# SAVE RESULT
# -----------------

output_csv = os.path.join(
    OUTPUT_FOLDER,
    "cluster_result.csv"
)

df.to_csv(
    output_csv,
    index=False
)

# -----------------
# SAVE REPORT
# -----------------

report_path = os.path.join(
    OUTPUT_FOLDER,
    "cluster_summary.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Silhouette Score : {score}\n\n"
    )

    f.write(
        str(
            df["cluster"]
            .value_counts()
            .sort_index()
        )
    )

print("DONE")
print(output_csv)
print(report_path)