from utils.data_preprocess import load_data, split_data, get_preprocessor
from model.logistic_regression import lr_model
from model.decision_tree import dt_model
from model.knn import knn_model
from model.naive_bayes import nb_model
from model.random_forest import rf_model
from model.xgboost_ensemble import xgb_model
from tabulate import tabulate


import pandas as pd

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

preprocessor = get_preprocessor(X_train)

results = [
    lr_model(X_train, X_test, y_train, y_test, preprocessor),
    dt_model(X_train, X_test, y_train, y_test, preprocessor),
    knn_model(X_train, X_test, y_train, y_test, preprocessor),
    nb_model(X_train, X_test, y_train, y_test, preprocessor),
    rf_model(X_train, X_test, y_train, y_test, preprocessor),
    xgb_model(X_train, X_test, y_train, y_test, preprocessor),
]

print(results)
df_results = pd.DataFrame(results)

print("\nModel Comparison Table\n")
print(tabulate(
    df_results,
    headers="keys",
    tablefmt="fancy_grid",
    floatfmt=".4f"
))