# Machine Learning Assignment - multiple classification models

## Problem Statement

Implement multiple ML classification models to a binary classification problem capturing all the metrics and deploy it on Streamlit Community Cloud.

---

## Dataset Description

### Adult (Census Income) Dataset

The Adult (Census Income) dataset contains demographic and employment-related information used to predict if a person earns more than $50K per year.

### Target Variable: `income`

The target variable represents whether a person earns more than $50K annually:
- **<=50K** - Earns $50,000 or less per year
- **>50K** - Earns more than $50,000 per year

This is a binary categorical variable used for classification.

### Features Description (14 Features)

#### 1. Demographic Features
- **age** – Continuous; age of the individual
- **sex** – Categorical; Male, Female
- **race** – Categorical; White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other
- **native-country** – Categorical; Country of origin

#### 2. Education Features
- **education** – Categorical; highest education level (e.g., Bachelors, HS-grad, Masters)
- **education-num** – Continuous; numerical representation of education level (numeric encoding of education)

#### 3. Employment Features
- **workclass** – Categorical; type of employer (Private, Self-emp, Government, etc.)
- **occupation** – Categorical; job type (Tech-support, Sales, Exec-managerial, etc.)
- **hours-per-week** – Continuous; number of working hours per week

#### 4. Financial Features
- **capital-gain** – Continuous; income from capital gains
- **capital-loss** – Continuous; losses from capital investments

#### 5. Social / Relationship Features
- **marital-status** – Categorical; marital condition (Married, Divorced, Never-married, etc.)
- **relationship** – Categorical; relationship role (Husband, Wife, Own-child, Not-in-family, etc.)
- **fnlwgt** – Continuous; final sampling weight assigned to individuals

---

## Model Comparison Results

Below is the comprehensive analysis of the machine learning models tested.

---

## Performance Metrics Overview

| S.No | Model | Accuracy | AUC | Precision | F1 | Recall | MCC |
|------|-------|----------|-----|-----------|----|----|-----|
| 1 | Logistic Regression | 0.8560 | 0.9098 | 0.7397 | 0.6745 | 0.6199 | 0.5867 |
| 2 | Decision Tree | 0.8174 | 0.7589 | 0.6151 | 0.6302 | 0.6460 | 0.5093 |
| 3 | K-Nearest Neighbors | 0.8365 | 0.8680 | 0.6745 | 0.6461 | 0.6199 | 0.5408 |
| 4 | Naive Bayes (Gaussian) | 0.5421 | 0.7498 | 0.3395 | 0.5008 | 0.9541 | 0.3341 |
| 5 | Random Forest | 0.8577 | 0.9077 | 0.7383 | 0.6818 | 0.6333 | 0.5937 |
| 6 | XGBoost | 0.8749 | 0.9302 | 0.7775 | 0.7214 | 0.6728 | 0.6440 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Strong baseline model with 85.6% accuracy and excellent AUC (0.9098). Demonstrates good balance between precision (0.7397) and recall (0.6199), making it reliable for general classification tasks with interpretable results. Serves as a strong benchmark for comparison with more complex models. |
| **Decision Tree** | Weakest performer among all models with 81.74% accuracy and lowest AUC (0.7589). Lower precision and F1 scores compared to other models suggest the model may be suffering from overfitting or lacks sufficient complexity to capture underlying patterns. Single decision trees typically struggle with generalization. |
| **K-Nearest Neighbors (kNN)** | Moderate performance with 83.65% accuracy and good AUC (0.8680). Balanced metrics across precision, recall, and F1 score indicate consistent but not exceptional predictive capability. Performance may vary significantly with different values of k parameter. |
| **Naive Bayes (Gaussian)** | Poorest accuracy at only 54.21% with very low precision (0.3395) but exceptionally high recall (0.9541). Aggressively predicts positive class, resulting in many false positives. Best suited for scenarios where minimizing false negatives is critical (e.g., disease screening), but not recommended for general balanced classification tasks. |
| **Random Forest (Ensemble)** | Excellent ensemble method and second-best overall with 85.77% accuracy and high AUC (0.9077). Demonstrates clear improvement over individual Decision Trees with balanced precision-recall trade-off. Robust performance through ensemble averaging reduces overfitting and improves generalization. |
| **XGBoost (Ensemble)** | ⭐ **BEST PERFORMING MODEL** ⭐ Achieves highest accuracy (87.49%), best AUC (0.9302), superior precision (0.7775), F1 score (0.7214), and MCC (0.6440). Advanced gradient boosting technique provides optimal predictive performance across all metrics. Recommended model for deployment. |

---

## Key Insights

1. **Winner:** XGBoost demonstrates clear superiority across all evaluation metrics
2. **Ensemble Advantage:** Both Random Forest and XGBoost (ensemble methods) significantly outperform individual models
3. **Naive Bayes Bias:** Shows extreme recall at the cost of precision, indicating strong bias toward positive class predictions
4. **Baseline Models:** Logistic Regression provides surprisingly competitive performance as a simple linear model
5. **MCC Analysis:** XGBoost has the highest Matthews Correlation Coefficient (0.6440), indicating the best balance between true/false positives and negatives

---

## Recommendation

**XGBoost** is recommended as the primary model for this classification task due to its superior performance across all metrics. Random Forest serves as an excellent alternative if model interpretability or faster training time is required.

---

*Model comparison completed successfully*
