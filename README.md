# Bank Customer Churn Prediction

An end-to-end machine learning pipeline for predicting credit card customer attrition — extended from a graduate school project into a deeper analytical exploration with real-world business decision framing.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Background

This project originated as a final assignment for a graduate machine learning course (MSSP 608), where the goal was to apply supervised and unsupervised learning techniques to a real-world dataset. The original submission covered exploratory data analysis, a baseline comparison of six classifiers, and PCA/K-Means clustering.

After completing the course, I extended the project significantly — going beyond the assignment scope to deepen the analysis and connect the modeling outputs to the kind of decisions a data science team would actually face in a banking context. The additions include domain-driven feature engineering, SHAP explainability for model transparency, silhouette-based cluster validation, and a cleaner narrative that ties each technical step to a concrete business question.

This repository reflects that ongoing work. I intend to continue building on it as my skills develop.

---

## Business Problem

Customer churn is one of the most costly challenges in retail banking. Acquiring a new credit card customer costs 5–7× more than retaining an existing one, which means identifying at-risk customers *before* they leave is far more valuable than reacting after the fact.

**Business questions this project addresses:**
- Which customers are most likely to close their credit card account in the next period?
- What behavioral signals best predict churn, and how early do they appear?
- Can we segment the customer base into actionable groups for targeted retention campaigns?
- Which model decisions are explainable enough to act on — and how do we communicate them to non-technical stakeholders?

**Best model:** Random Forest — ~96% accuracy, ~0.87 Cohen's Kappa on held-out test data.

---

## Dataset

- **Source:** [Credit Card Customers — Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **Size:** 10,127 customers × 21 features (after cleaning)
- **Target:** `Attrition_Flag` — binary churn indicator (16% positive class)
- **Features:** Customer demographics (age, gender, education, income) + behavioral metrics (transaction counts, transaction amounts, revolving balance, inactivity months)

---

## Project Structure

```
bank-churn-prediction/
├── bank_churn_prediction.ipynb   # Main analysis notebook
├── BankChurners.csv              # Raw dataset (download from Kaggle)
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## Methodology

### 1 · Exploratory Data Analysis
- Distribution analysis with churn-stratified KDE plots
- IQR-based outlier detection
- Correlation heatmap to surface the strongest predictors of churn

### 2 · Feature Engineering
The original dataset had informative raw features, but several higher-signal interactions were left unexploited. I created five domain-driven features to capture patterns a banker would actually reason about:

| Feature | Formula | Business Intuition |
|:--------|:--------|:----------|
| `spend_per_txn` | `Total_Trans_Amt / Total_Trans_Ct` | Declining average spend per visit signals disengagement |
| `balance_utilization` | `Total_Revolving_Bal / Credit_Limit` | Low utilization on a high-limit card may indicate the customer is moving spend elsewhere |
| `txn_activity_change` | `Total_Trans_Ct × Total_Ct_Chng_Q4_Q1` | Combines volume with trend — a customer with falling volume *and* a downward trend is high risk |
| `contacts_per_month_inactive` | `Contacts_Count_12_mon / (Months_Inactive_12_mon + 1)` | High contact frequency normalized by inactivity can flag customers who are dissatisfied but still engaging with support |
| `credit_per_dependent` | `Credit_Limit / (Dependent_count + 1)` | Household credit load — a proxy for financial stress |

### 3 · Supervised Learning
Six classifiers evaluated across 8 configurations:
- **Algorithms:** Naïve Bayes, Logistic Regression, SVM, Decision Tree, Random Forest, AdaBoost
- **Evaluation:** Accuracy + Cohen's Kappa across baseline splits, 5-fold CV, 10-fold CV, stratified CV, feature-subset CV, and two hyperparameter tuning rounds
- **Visualizations:** Confusion matrices, ROC/AUC curves, feature importance, SHAP explainability

SHAP values were added specifically to address the business need for interpretability — a model that's 96% accurate but unexplainable is often unusable in regulated industries like banking.

### 4 · Dimensionality Reduction (PCA)
- Scree plot + cumulative variance analysis
- PC loading interpretation to understand which original features drive each component
- RF accuracy vs. number of components to find the compression/performance tradeoff

### 5 · Unsupervised Learning (K-Means)
- Optimal K selection via Elbow Method + Silhouette Score (the original assignment only used the elbow method — silhouette scoring adds a more rigorous validation)
- Cluster profiling by churn rate and feature means
- 2D PCA projection of clusters for visual interpretation

---

## Key Results

| Model | Accuracy | Cohen's Kappa |
|:------|:--------|:-------------|
| Random Forest | ~96% | ~0.87 |
| AdaBoost | ~95% | ~0.85 |
| Decision Tree | ~93% | ~0.78 |
| SVM | ~92% | ~0.75 |
| Logistic Regression | ~91% | ~0.72 |
| Naïve Bayes | ~87% | ~0.59 |

**Top churn predictors:** `Total_Trans_Ct`, `Total_Trans_Amt`, `Total_Ct_Chng_Q4_Q1`, `Contacts_Count_12_mon`, `spend_per_txn` (engineered)

Cohen's Kappa is reported alongside accuracy because the dataset is class-imbalanced (16% churn) — a model that predicts "no churn" every time would achieve 84% accuracy but a Kappa of 0.

---

## Business Recommendations

The model outputs translate directly into retention actions:

- **Early warning triggers:** Flag customers whose `Total_Trans_Ct` drops >20% quarter-over-quarter for proactive outreach — this feature is the single strongest churn predictor and is observable in near-real-time.
- **High-value at-risk segment:** Customers with high credit limits but low and declining utilization are disproportionately represented in the churn group. These are often the most valuable customers to retain — prioritize them for personalized offers.
- **Segment-level campaigns:** K-Means clusters reveal distinct behavioral profiles. Rather than a one-size-fits-all retention strategy, cluster labels can be used to tailor messaging — e.g., transaction reactivation offers for low-activity clusters vs. fee waiver campaigns for high-contact, high-inactivity segments.
- **Model transparency for stakeholders:** SHAP waterfall plots make individual predictions auditable, which matters for customer-facing decisions in regulated environments.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/nina-pham/bank-churn-prediction.git
cd bank-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place BankChurners.csv in the project root
# Then open the notebook
jupyter notebook bank_churn_prediction.ipynb
```

---

## License

MIT License — free to use and adapt with attribution.
