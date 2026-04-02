# Rain in Australia — ML Prediction Project

Supervised machine learning project to predict whether it will rain tomorrow in Australia using historical weather data. Four distinct algorithms are explored and compared.

---

## Project Structure

| File | Description |
|------|-------------|
| `weatherAUS.csv` | Original, imbalanced dataset (~145K samples) |
| `weatherAUS_balanced.csv` | Class-balanced dataset (220,632 samples, 50/50 split) |
| `Dataset_balancing.ipynb` | Data exploration and SMOTE-based class balancing |
| `01_logistic_regression_weatherAUS.ipynb` | Logistic Regression model |
| `02_decision_tree_weatherAUS.ipynb` | Decision Tree model |
| `03_random_forest_weatherAUS.ipynb` | Random Forest model |
| `04_gradient_boosting_weatherAUS.ipynb` | Gradient Boosting model |

---

## Dataset

- **Source:** [Kaggle – Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
- **Original size:** ~145,000 daily weather observations across multiple Australian stations
- **Balanced size:** 220,632 rows (110,316 per class)
- **Target variable:** `RainTomorrow` (binary: 0 = No Rain, 1 = Rain)
- **Features (23 columns):** Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, Wind direction & speed, Humidity, Pressure, Cloud cover, Temperature, RainToday

### Key Data Challenges
- Severe class imbalance (~77% No Rain, ~23% Rain) — resolved via oversampling/balancing
- High missing value rates: Sunshine (47.5%), Evaporation (43.2%), Cloud9am (36.9%), Cloud3pm (38.8%)
- Mixed numeric and categorical features

---

## ML Pipeline (Common Steps)

All notebooks follow a similar workflow:

1. **Load & inspect** the balanced dataset
2. **Clean** target variable, handle `RainToday` encoding
3. **Feature engineering** — extract Year, Month, DayOfYear from Date
4. **Preprocessing** — median imputation for numeric, mode/one-hot encoding for categorical
5. **Train/test split** — 80/20, stratified
6. **Train** the model
7. **Evaluate** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

---

## Models & Results

### 1. Logistic Regression (`01_logistic_regression_weatherAUS.ipynb`)

- **Solver:** `liblinear`, C=1.0, max_iter=500
- **Feature selection:** `SelectKBest` (top 20 features, ANOVA F-test)
- **Preprocessing:** StandardScaler + SimpleImputer + OneHotEncoder

| Metric | Score |
|--------|-------|
| Accuracy | **78.1%** |
| Precision (Rain) | 0.79 |
| Recall (Rain) | 0.77 |
| F1-Score | 0.78 |
| ROC-AUC | **0.865** |

---

### 2. Decision Tree (`02_decision_tree_weatherAUS.ipynb`)

- **Config:** max_depth=10, min_samples_split=20, min_samples_leaf=10, class_weight='balanced'
- **Preprocessing:** Median imputation + LabelEncoding for categoricals
- **Note:** Dropped high-missingness columns (Sunshine, Evaporation, Cloud9am, Cloud3pm)

| Metric | Score |
|--------|-------|
| Accuracy | **78.7%** |
| Precision (Rain) | 0.786 |
| Recall (Rain) | 0.788 |
| F1-Score | 0.787 |
| ROC-AUC | **0.870** |

---

### 3. Random Forest (`03_random_forest_weatherAUS.ipynb`)

- **Preprocessing:** Median imputation (numeric) + OneHotEncoder (categorical)
- **Includes:** Training vs. testing accuracy analysis and feature importance visualization

| Metric | Score |
|--------|-------|
| Accuracy (Test) | **88.8%** |
| Training Accuracy | 93.1% |

---

### 4. Gradient Boosting (`04_gradient_boosting_weatherAUS.ipynb`)

Two variants compared — baseline and hyperparameter-tuned:

| Metric | Baseline | Tuned |
|--------|----------|-------|
| Accuracy | 79% | **81%** |
| ROC-AUC | 0.878 | **0.895** |

---

## Model Comparison Summary

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 78.1% | 0.865 |
| Decision Tree | 78.7% | 0.870 |
| Gradient Boosting (Tuned) | 81.0% | 0.895 |
| **Random Forest** | **88.8%** | — |

> **Best overall model:** Random Forest with 88.8% test accuracy.

---

## 🛠️ Requirements

All notebooks were developed in **Google Colab** and use:

```
scikit-learn
pandas
numpy
matplotlib
```

> **Note:** Upload `weatherAUS_balanced.csv` to `/content/` in Colab before running the notebooks (or adjust the `DATA_PATH` variable).

---

## Notes

- Deep learning models were intentionally excluded (assignment constraint)
- Class balancing was performed as a preprocessing step using the `Dataset_balancing.ipynb` notebook
- Each notebook is self-contained and can be run independently