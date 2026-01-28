from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from utils.data_preprocess import load_preprocess
from utils.model_evaluate import print_metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)


def xgb_model(X_train, X_test, y_train, y_test, preprocessor):
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "Model Name": "XGBoost",
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
