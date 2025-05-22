# 🌳 Decision Trees Mastery: From Basic to Advanced

Welcome to the ultimate repository to master **Decision Trees** in Machine Learning — from theory to implementation, along with extensive practice across real-world datasets.

---

## 📘 What is a Decision Tree?

A Decision Tree is a supervised learning algorithm used for classification and regression. It models decisions as a tree structure, splitting the dataset into subsets based on feature values.

---

## 🧠 Core Concepts & Math

### 🔹 Entropy (Information Gain)
  `Entropy(S) = - Σ (p_i * log2(p_i))`  
  where `p_i` is the probability of class `i` in dataset `S`.

- **Information Gain**:  
  It is the reduction in entropy after a dataset is split using a feature.

### 🔹 Gini Impurity
  `Gini(S) = 1 - Σ (p_i)^2`  
  where `p_i` is the probability of class `i` in dataset `S`.

### 🔹 Overfitting in Trees
- Happens when a tree is too deep and memorizes training data.
- Symptoms: High training accuracy, poor test performance.

### 🔹 Pruning
- Reduces tree size to prevent overfitting.
- **Pre-Pruning**: Limit `max_depth`, `min_samples_split`, etc.
- **Post-Pruning**: Use `ccp_alpha` (Cost Complexity Pruning)

---

## 📂 Projects

### 🌱 BASIC LEVEL
| Project | Dataset | Concepts |
|--------|---------|----------|
| ✅ Play Cricket Classifier | Manual | Manual Entropy/Gini |
| ✅ Iris Flower Classification | `sklearn.datasets.load_iris` | Gini, Entropy |
| ✅ Titanic Survival Prediction | Kaggle Titanic / Seaborn | Categorical handling, missing values |

---

### 🌿 MEDIUM LEVEL
| Project | Dataset | Concepts |
|--------|---------|----------|
| ✅ Loan Default Prediction | Lending Club / UCI Credit | Imbalanced data, post-pruning |
| ✅ Income Classification | UCI Adult Income | Encoding, feature importance |
| ✅ Breast Cancer Diagnosis | `sklearn.datasets.load_breast_cancer` | Tree visualization, pruning |
| ✅ Student Performance | UCI Student Dataset | Feature engineering, domain logic |

---

### 🌳 ADVANCED LEVEL
| Project | Dataset | Concepts |
|--------|---------|----------|
| ✅ Flight Fare Category Prediction | Kaggle Flight Fare | Feature extraction from dates |
| ✅ E-commerce Product Classification | Custom / Amazon | Text + structured data |
| ✅ Medical Cost Risk Classification | Insurance Dataset | Regression + classification |
| ✅ Heart Disease Prediction | UCI Heart | ROC curve, hyperparameter tuning |
| ✅ Customer Churn Prediction | IBM / Kaggle | Class imbalance, cross-validation |

---

## 🛠️ Tools & Libraries

- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `sklearn.tree.plot_tree`, `plotly`
- `GridSearchCV`, `cross_val_score`

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/decision-trees-mastery.git
   cd decision-trees-mastery
