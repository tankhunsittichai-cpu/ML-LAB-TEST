import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from data_loader import load_data
from knn_tf import build_model
from evaluate import evaluate_model

# =====================
# PATH
# =====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "age_gender(1).csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output_folder"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Start Program")
print("CSV Path =", CSV_PATH)

# =====================
# LOAD DATA
# =====================

X, y = load_data(CSV_PATH)

print("Load Success")

# =====================
# SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================
# TRAIN
# =====================

model = build_model()

model.fit(X_train, y_train)

print("Train Success")

# =====================
# PREDICT
# =====================

y_pred = model.predict(X_test)

print("Predict Success")

# =====================
# EVALUATE
# =====================

acc, report = evaluate_model(
    y_test,
    y_pred
)

print("Accuracy =", acc)

# =====================
# SAVE MODEL
# =====================

joblib.dump(
    model,
    os.path.join(
        OUTPUT_DIR,
        "model.pkl"
    )
)

# =====================
# SAVE CSV
# =====================

result_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
})

result_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    ),
    index=False
)

# =====================
# SAVE REPORT
# =====================

with open(
    os.path.join(
        OUTPUT_DIR,
        "result.txt"
    ),
    "w"
) as f:

    f.write(f"Accuracy : {acc}\n")
    f.write(report)

print("Done!")