# Decision Tree Classification & Regression

A comprehensive implementation of Decision Tree models using Python and `scikit-learn`. This project demonstrates data preprocessing, model training, hyperparameter tuning, tree visualization, and evaluation metrics.

---

## 📌 Overview

Decision Trees are non-parametric supervised learning methods used for both classification and regression. The model predicts the value of a target variable by learning simple decision rules inferred from the data features.

### Key Objectives:
- Implement `DecisionTreeClassifier` and `DecisionTreeRegressor`.
- Evaluate splitting criteria (Gini Impurity, Entropy / Log Loss, Mean Squared Error).
- Address and prevent overfitting using pruning techniques (`max_depth`).
- Visualize tree architecture and feature importances.

---

## ⚙️ Core Concepts

* **Splitting Criteria:**
  * **Gini Impurity:** Measures the frequency at which any element of the dataset will be mislabelled when randomly labeled.
  * **Entropy / Information Gain:** Quantifies the uncertainty reduction after a split.
  * **MSE / MAE (Regression):** Minimizes variance or absolute deviations within terminal nodes.
* **Overfitting Control (Pruning):**
  * **Pre-Pruning:** Setting early-stopping constraints such as `max_depth`, `min_samples_split`, and `min_samples_leaf`.
  * **Post-Pruning:** Cost-Complexity Pruning via `ccp_alpha` to trim weak branches after the full tree is built.

---

## 🛠️ Tech Stack & Requirements

* **Python 3.8+**
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib` / `seaborn`
* `graphviz` (optional, for advanced tree rendering)

Install dependencies:
```bash
pip install -r requirements.txt
