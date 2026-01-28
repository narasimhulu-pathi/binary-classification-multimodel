import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
from pathlib import Path


# def load_preprocess(path="/Users/narasimhulupathi/PycharmProjects/pyTorch/binary-classification-multimodel/data"
#                          "/adult_income.csv"):
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

def load_data(path="data/adult.csv"):
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
    return df

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
