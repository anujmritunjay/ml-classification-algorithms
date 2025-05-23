# 🌳 Mastering Random Forest: From Beginner to Expert

This repository is your **complete guide to learning Random Forest**, covering everything from theory and math to hands-on projects at every level.

---

## 📚 Table of Contents

* [🔍 Introduction](#-introduction)
* [🧠 What is Random Forest?](#-what-is-random-forest)
* [🖐️ Math Behind Random Forest](#-math-behind-random-forest)
* [🧺 Bagging (Bootstrap Aggregation)](#-bagging-bootstrap-aggregation)
* [🔀 Random Feature Selection](#-random-feature-selection)
* [🔥 Feature Importance](#-feature-importance)
* [⚙️ Hyperparameters](#%ef%b8%8f-hyperparameters)
* [📊 Evaluation Metrics](#-evaluation-metrics)
* [🚀 Projects for Practice](#-projects-for-practice)
* [🧪 Advanced Concepts](#-advanced-concepts)
* [🗓 Learning Schedule](#-learning-schedule)
* [📦 Setup & Installation](#-setup--installation)
* [🙌 Contribute](#-contribute)
* [📜 License](#-license)
* [📢 Contact](#-contact)

---

## 🔍 Introduction

**Random Forest** is a powerful ensemble machine learning algorithm that builds multiple decision trees and combines their predictions for better accuracy and stability. It is used for both **classification** and **regression** tasks.

---

## 🧠 What is Random Forest?

Random Forest is an ensemble of Decision Trees that works by:

* Creating multiple Decision Trees using **bootstrapped samples** of the training data (Bagging)
* At each split, selecting a **random subset of features** (Random Subspace Method)
* Combining predictions of all trees:

  * For classification: majority vote
  * For regression: average prediction

---

## 🖐️ Math Behind Random Forest

### 1. Gini Impurity (for Classification)

Used to measure the impurity of a node in decision trees:

```
Gini(t) = 1 - sum(p_i^2)
```

* `t` is the node
* `p_i` is the proportion of samples of class `i` in node `t`
* Lower Gini means purer nodes

---

### 2. Mean Squared Error (for Regression)

Used as a loss function to determine the quality of a split in regression:

```
MSE = (1/n) * sum((y_i - y_mean)^2)
```

* `y_i` is the actual value of the target
* `y_mean` is the average prediction of the node
* `n` is the number of samples

---

### 3. Bagging (Bootstrap Aggregation)

Given a dataset `D`, we:

* Generate `B` random bootstrap samples: `D1`, `D2`, ..., `DB`
* Train `B` trees: `T1`, `T2`, ..., `TB`
* For prediction:

  * **Classification**: take the majority vote from all trees
  * **Regression**: take the average of all tree predictions

Bagging helps reduce overfitting and increases generalization.

---

### 4. Random Feature Selection

At each split in each tree:

* Select a random subset of `m` features from the total `M` features (`m < M`)
* Choose the best split only from those `m` features
* This increases diversity among trees and reduces correlation

---

## 🧺 Bagging (Bootstrap Aggregation)

* Trains each tree on a random sample **with replacement**
* This creates diverse trees
* Predictions are combined to reduce variance and improve robustness

---

## 🔀 Random Feature Selection

* Instead of using all features at each node, Random Forest selects a **random subset**
* Helps in reducing overfitting
* Makes each tree less similar, improving the ensemble performance

---

## 🔥 Feature Importance

Random Forest ranks features by how much they improve the model's performance.

**How it works:**

* When a feature is used to split a node, the impurity is reduced
* These reductions are averaged and accumulated across all trees

You can access feature importances using:

```python
model.feature_importances_
```

---

## ⚙️ Hyperparameters

| Hyperparameter      | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `n_estimators`      | Number of trees in the forest                        |
| `max_depth`         | Maximum depth of each tree                           |
| `max_features`      | Number of features to consider per split             |
| `min_samples_split` | Minimum samples required to split an internal node   |
| `min_samples_leaf`  | Minimum samples required at a leaf node              |
| `bootstrap`         | Whether to use bootstrap samples when building trees |
| `oob_score`         | Use out-of-bag samples to estimate accuracy          |

---

## 📊 Evaluation Metrics

### For Classification

* Accuracy
* Precision, Recall, F1 Score
* Confusion Matrix
* ROC AUC Score

### For Regression

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

---

## 🚀 Projects for Practice

### ✅ Beginner Projects

* [x] Iris Flower Classification
* [x] Titanic Survival Prediction
* [x] Wine Quality Prediction

### 📈 Intermediate Projects

* [x] Medical Insurance Cost Prediction
* [ ] Loan Default Classification
* [ ] Employee Attrition Prediction

### 💼 Advanced Projects

* [ ] Flight Ticket Price Prediction
* [ ] Credit Card Fraud Detection (Imbalanced Dataset)
* [ ] Customer Segmentation (Clustering + Random Forest)

---

## 🧪 Advanced Concepts

* **Out-of-Bag (OOB) Score**: Evaluate model without separate validation set
* **Feature Importance**: Based on impurity decrease or permutation
* **GridSearchCV**: Find optimal hyperparameters
* **Handling Imbalanced Classes**: Use class weights, SMOTE, etc.
* **Explainability**: Use SHAP or LIME to interpret predictions

---

## 🗓 Learning Schedule

| Week | Focus Area                                 | Activities               |
| ---- | ------------------------------------------ | ------------------------ |
| 1    | Basics of Decision Trees and Random Forest | Iris, Titanic projects   |
| 2    | Bagging, Feature Importance, Parameters    | Wine, Insurance projects |
| 3    | Model Tuning, Imbalanced Data              | Loan, Fraud projects     |
| 4    | Explainability, SHAP, Project Work         | Flights, Segmentation    |

---


### Required Libraries

```
scikit-learn
pandas
numpy
matplotlib
seaborn
shap
```

---

## 🙌 Contribute

* Fork the repository
* Create a new branch
* Commit your changes
* Submit a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 📢 Contact

Connect with me on:

* GitHub: [@anujmritunjay](https://github.com/anujmritunjay)
