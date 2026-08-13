import pandas as pd
import numpy as np


def load_data(csv_path):

    df = pd.read_csv(csv_path)

    X = np.array([
        np.fromstring(
            pixel,
            sep=" "
        )
        for pixel in df["pixels"]
    ])

    return X, df