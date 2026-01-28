from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)


def nb_model(X_train, X_test, y_train, y_test, preprocessor):
    X_train_nb = preprocessor.fit_transform(X_train).toarray()
    X_test_nb = preprocessor.transform(X_test).toarray()

    model = GaussianNB()
    model.fit(X_train_nb, y_train)

    y_pred = model.predict(X_test_nb)
    y_prob = model.predict_proba(X_test_nb)[:, 1]

    return {
        "Model Name": "Naive Bayes (Gaussian)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
