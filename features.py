import numpy as np
import pandas as pd

RAW_FEATURES = ["Submission.Actions", "Quiz.Actions", "Forum.Actions", "Create.Actions",
                "Read.Actions", "Update.Actions", "Delete.Actions", "Days.Logged.In"]
TARGET = "rating"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    X = df[RAW_FEATURES].astype(float).clip(lower=0).copy()
    days = X["Days.Logged.In"].replace(0, np.nan)
    X["Total.Actions"] = X[RAW_FEATURES[:-1]].sum(axis=1)
    X["Write.Actions"] = X[["Submission.Actions", "Create.Actions", "Update.Actions", "Delete.Actions"]].sum(axis=1)
    X["Actions.Per.Day"] = (X["Total.Actions"] / days).fillna(0)
    X["Reads.Per.Day"] = (X["Read.Actions"] / days).fillna(0)
    X["Write.Read.Ratio"] = X["Write.Actions"] / (X["Read.Actions"] + 1)
    X["Uses.Quiz"] = (X["Quiz.Actions"] > 0).astype(float)
    X["Uses.Forum"] = (X["Forum.Actions"] > 0).astype(float)
    # log1p tames extreme outliers (Read.Actions max ~399k)
    for c in RAW_FEATURES + ["Total.Actions", "Write.Actions", "Actions.Per.Day", "Reads.Per.Day"]:
        X["log_" + c] = np.log1p(X[c])
    return X
