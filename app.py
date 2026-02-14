from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from utils.data_preprocess import load_data, split_data, get_preprocessor
from model.logistic_regression import lr_model
from model.decision_tree import dt_model
from model.knn import knn_model
from model.naive_bayes import nb_model
from model.random_forest import rf_model
from model.xgboost_ensemble import xgb_custom_model

# Page config
st.set_page_config(
    page_title="Adult Income – Model Comparison",
    layout="wide"
)

st.title("Adult Income Dataset – ML Classification Models Comparison")
st.caption("Compare multiple classification models with metrics and diagnostics")

# Model registry
MODEL_MAP = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "K-Nearest Neighbors": knn_model,
    "Naive Bayes (Gaussian)": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_custom_model
}

MODEL_NAMES = list(MODEL_MAP.keys())


# Caching (important)


@st.cache_resource
def build_preprocessor(X_train):
    return get_preprocessor(X_train)


# Sidebar – Fancy selector
st.sidebar.markdown("Model Playground")
st.sidebar.caption("Toggle models to include/exclude")
st.sidebar.divider()

# Session state initialization
if "select_all" not in st.session_state:
    st.session_state.select_all = True

for name in MODEL_NAMES:
    if name not in st.session_state:
        st.session_state[name] = True  # each toggle


# Select All checkbox (derived state)
def on_select_all_change():
    for name in MODEL_NAMES:
        st.session_state[name] = st.session_state.select_all


# Compute select_all from individual toggles
all_selected = all(st.session_state[name] for name in MODEL_NAMES)
st.session_state.select_all = all_selected

st.sidebar.checkbox(
    "Select all models",
    key="select_all",
    on_change=on_select_all_change
)

# Individual toggles (ordered)
selected_models = []

for name in MODEL_NAMES:
    st.sidebar.toggle(name, key=name)
    if st.session_state[name]:
        selected_models.append(name)

# Safety fallback: none selected → auto-select all
if not selected_models:
    for name in MODEL_NAMES:
        st.session_state[name] = True
    st.session_state.select_all = True
    selected_models = MODEL_NAMES.copy()

# File upload section
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file for testing (optional)",
    type=["csv"]
)

if uploaded_file:
    st.sidebar.success("File uploaded successfully!")


@st.cache_data
def load_and_split_data(uploaded_file=None):
    df = load_data(uploaded_file)
    return split_data(df)


# Run models
if st.button("Run Selected Models for Comparison"):

    with st.spinner("Training models and computing metrics..."):

        X_train, X_test, y_train, y_test = load_and_split_data(uploaded_file)
        preprocessor = build_preprocessor(X_train)

        results = []
        model_outputs = {}

        for name in selected_models:
            output = MODEL_MAP[name](
                X_train, X_test, y_train, y_test, preprocessor
            )

            model_outputs[name] = output

            results.append({
                "Model": output["Model Name"],
                "Accuracy": output["Accuracy"],
                "AUC": output["AUC"],
                "Precision": output["Precision"],
                "F1": output["F1"],
                "Recall": output["Recall"],
                "MCC": output["MCC"]
            })

        df_results = (
            pd.DataFrame(results)
            # .sort_values("AUC", ascending=False)
            # .reset_index(drop=True)
        )

    st.success("Model comparison completed")

    # Comparison table
    st.markdown("Models Comparison Table")

    # Add serial number column starting from 1
    df_display = df_results.copy()
    df_display.insert(0, "S.No", range(1, len(df_display) + 1))

    numeric_cols = df_display.select_dtypes(include="number").columns

    st.dataframe(
        df_display.style.format(
            {c: "{:.4f}" for c in numeric_cols if c != "S.No"}
        ),
        width="stretch",
        hide_index=True
    )

    # Download CSV
    csv = df_display.round(4).to_csv(index=False).encode("utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"model_comparison_{timestamp}.csv"

    st.download_button(
        "Download metrics as CSV",
        data=csv,
        file_name=file_name,
        mime="text/csv"
    )

    # Diagnostics
    st.markdown("Model Diagnostics")

    for name, output in model_outputs.items():
        with st.expander(f" {name}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(
                    output["confusion_matrix"]
                ).plot(ax=ax)
                st.pyplot(fig)

            with col2:
                st.subheader("ROC Curve")
                fig, ax = plt.subplots()
                RocCurveDisplay.from_predictions(
                    output["y_test"],
                    output["y_proba"],
                    ax=ax
                )
                st.pyplot(fig)

else:
    st.info("👈 Select models in the sidebar and click **Run Model Comparison**")
