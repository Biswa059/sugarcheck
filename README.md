# 🩺 SugarCheck — Early Diabetes Risk Screening

SugarCheck is a machine-learning based diabetes risk screening application.
It predicts the probability of diabetes using clinical and demographic
features from the Pima Indians Diabetes Dataset.

> ⚠️ SugarCheck is a screening-support tool, not a medical diagnosis.

---

## 🚀 Features

- Data auditing and exploratory data analysis
- Hidden missing-value detection
- Feature engineering
- Median imputation using ML pipelines
- Standardization for scale-sensitive models
- Comparison of 6 classification models
- Stratified 5-fold cross-validation
- Hyperparameter tuning using RandomizedSearchCV
- ROC-AUC based model selection
- Threshold tuning to prioritize diabetic-class recall
- ROC and Precision-Recall curves
- Confusion matrix analysis
- Logistic Regression feature importance
- Streamlit web application

---

## 🤖 Models Compared

The following models were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. XGBoost
6. Support Vector Machine

### Final Model

**Logistic Regression**

Best tuned CV ROC-AUC:

**0.8454**

---

## 📊 Final Test Performance

The final model was evaluated once on the unseen test set.

| Metric | Score |
|---|---:|
| Accuracy | 70.78% |
| Precision | 55.42% |
| Recall | 85.19% |
| F1 Score | 67.15% |
| ROC-AUC | 81.37% |

### Confusion Matrix

```text
[[63 37]
 [ 8 46]]