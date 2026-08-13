import pandas as pd
import numpy as np

def load_data(csv_path):

    df = pd.read_csv(csv_path)

    print(df.head())

    X = df["pixels"].apply(
        lambda x: np.array(
            list(map(int, x.split()))
        )
    )

    X = np.vstack(X.values)

    y = df["gender"]

    return X, y