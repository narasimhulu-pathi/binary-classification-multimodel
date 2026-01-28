from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)


def print_metrics(name, y_true, y_pred, y_prob):
    print(f"--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("AUC      :", roc_auc_score(y_true, y_prob))
    print("MCC      :", matthews_corrcoef(y_true, y_pred))
    print()
