import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
from pathlib import Path


def load_preprocess(path="./data/adult_income.csv"):
    # Folder where script is
    script_dir = Path(__file__).parent

    # Path to CSV relative to script
    csv_path = script_dir.parent / "data" / "adult_income.csv"
    print(os.getcwd())  # Current working directory
    print(os.path.exists(path))  # Does the file exist?
    df = pd.read_csv(csv_path)

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

    X = df.drop("income", axis=1)
    y = df["income"]

    numeric = X.select_dtypes(include=["int64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    ), preprocessor


# Expected column names in order
DEFAULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]


def load_data(uploaded_file=None):
    """
    Load dataset from uploaded CSV (with or without headers) or default CSV.
    Throws exception if CSV shape/order is incorrect.
    """
    try:
        if uploaded_file is not None:
            # Assume no headers
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] != len(DEFAULT_COLUMNS):
                raise ValueError(
                    f"Uploaded CSV has {df.shape[1]} columns, "
                    f"but expected {len(DEFAULT_COLUMNS)} columns in this order: {DEFAULT_COLUMNS}"
                )
            df.columns = DEFAULT_COLUMNS
        else:
            # Default dataset (has headers)
            script_dir = Path(__file__).parent
            csv_path = script_dir.parent / "data" / "adult_income.csv"
            df = pd.read_csv(csv_path, header=0)
            if list(df.columns) != DEFAULT_COLUMNS:
                raise ValueError(
                    f"Default CSV columns do not match expected order:\nExpected: {DEFAULT_COLUMNS}\nGot: {list(df.columns)}"
                )

        # Preprocessing
        df.replace("?", np.nan, inplace=True)
        df.dropna(inplace=True)
        df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

        return df

    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")


def get_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])


def split_data(df):
    X = df.drop("income", axis=1)
    y = df["income"]

    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
