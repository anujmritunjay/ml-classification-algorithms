# 🧠 K-Nearest Neighbors (KNN) Algorithm for Classification

This repository is dedicated to mastering the **K-Nearest Neighbors (KNN)** algorithm for **classification**, from very basic to most advanced levels. It includes in-depth **mathematical intuition**, **theoretical insights**, and a structured list of **hands-on projects** to build expertise through real-world data.

---

## 📌 Table of Contents

- [🔍 What is KNN?](#-what-is-knn)
- [🧠 Intuition Behind KNN](#-intuition-behind-knn)
- [📐 Mathematical Foundation](#-mathematical-foundation)
- [🧮 Distance Metrics](#-distance-metrics)
- [🔢 Choosing the Value of K](#-choosing-the-value-of-k)
- [⚖️ Feature Scaling](#️-feature-scaling)
- [📉 Curse of Dimensionality](#-curse-of-dimensionality)
- [🔁 Weighted KNN (Advanced)](#-weighted-knn-advanced)
- [🧪 Evaluation Metrics](#-evaluation-metrics)
- [🚀 Real-World Projects](#-real-world-projects)
- [📊 Capstone Project Ideas](#-capstone-project-ideas)
- [🗂️ Learning Plan](#️-learning-plan)

---

## 🔍 What is KNN?

**K-Nearest Neighbors (KNN)** is a simple, intuitive, non-parametric, **supervised learning algorithm** used for classification and regression.

- **Instance-based**: No training phase; all work happens during prediction.
- **Lazy learner**: It stores all training data and makes predictions based on proximity.

---

## 🧠 Intuition Behind KNN

Imagine moving to a new city and wanting to guess the profession of your neighbor. You ask the 5 nearest people:

- 3 are doctors
- 2 are engineers

You predict your neighbor is likely a **doctor**. This is the core idea of KNN: classify a point based on the **majority class of its K nearest neighbors**.

---

## 📐 Mathematical Foundation

Given a query point `x₀`, the KNN algorithm:

1. Calculates the distance between `x₀` and all points in the training set.
2. Selects the `K` nearest neighbors to `x₀`.
3. Takes a majority vote (in classification) from the labels of these K neighbors.

---

## 🧮 Distance Metrics

KNN depends heavily on how distance is measured.

### 1. Euclidean Distance

    d(x, y) = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2)

### 2. Manhattan Distance

    d(x, y) = |x1 - y1| + |x2 - y2| + ... + |xn - yn|

### 3. Minkowski Distance

    d(x, y) = (|x1 - y1|^p + |x2 - y2|^p + ... + |xn - yn|^p)^(1/p)

    - p = 1 → Manhattan distance
    - p = 2 → Euclidean distance

### 4. Cosine Similarity (for text data and high dimensions)

    cos(θ) = (A · B) / (||A|| * ||B||)

Where:
- A · B is the dot product of vectors A and B
- ||A|| is the magnitude of vector A

---

## 🔢 Choosing the Value of K

- Small K → flexible, may overfit (low bias, high variance)
- Large K → more stable, may underfit (high bias, low variance)
- Use **odd values** of K for binary classification to avoid ties.
- Use **cross-validation** to find the best K value.

---

## ⚖️ Feature Scaling

KNN is sensitive to the scale of features. Distance can be dominated by features with larger ranges.

**Apply scaling methods before training:**

### Standardization (Z-score normalization):

    z = (x - mean) / standard_deviation

### Min-Max Normalization:

    x_scaled = (x - min) / (max - min)

---

## 📉 Curse of Dimensionality

As the number of features (dimensions) increases:

- All distances tend to become similar
- KNN struggles to find meaningful neighbors
- Computation cost increases significantly

**Solution:** Reduce dimensionality using PCA or feature selection.

---

## 🔁 Weighted KNN (Advanced)

Instead of equal weight, assign more importance to **closer neighbors**.

### Inverse Distance Weighting:

    Weight of neighbor i = 1 / (distance(x, xi)^2)

This improves accuracy when closer neighbors are more reliable.

---

## 🧪 Evaluation Metrics

Common classification metrics:

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall (Sensitivity)** = TP / (TP + FN)
- **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)
- **Confusion Matrix**: Table showing TP, TN, FP, FN
- **ROC Curve and AUC**: For binary classifiers

---

## 🚀 Real-World Projects

| Project # | Title |
|----------:|-------|
| 1 | Iris Flower Classification |
| 2 | Breast Cancer Detection |
| 3 | Titanic Survival Classification |
| 4 | Handwritten Digits Recognition |
| 5 | Wine Quality Prediction |
| 6 | Diabetes Risk Detection |
| 7 | Heart Disease Classification |
| 8 | Bank Customer Churn Classification |
| 9 | Loan Approval Prediction |
| 10 | Voice Gender Recognition |
| 11 | Spam vs. Ham Email Classification |
| 12 | Text Document Classification |
| 13 | Customer Segmentation with PCA + KNN |

---

## 📊 Capstone Project Ideas

| # | Project Idea |
|--:|--------------|
| 1 | Credit Card Fraud Detection |
| 2 | Fake News Detection (TF-IDF + KNN) |
| 3 | Facial Expression Recognition |
| 4 | Pneumonia Detection via Image Features |
| 5 | Real-time Traffic Sign Classification |
| 6 | Breast Cancer Classifier with Deployment |
| 7 | Skin Disease Classification from Symptoms |

---

## 🗂️ Learning Plan

| Day | Focus |
|-----|-------|
| 1-2 | Learn KNN basics and intuition |
| 3-5 | Understand distance metrics and feature scaling |
| 6-7 | Learn evaluation metrics (Accuracy, Precision, etc.) |
| 8-12 | Complete beginner to intermediate projects |
| 13-15 | Explore weighted KNN and advanced use-cases |
| 16-18 | Learn dimensionality reduction with PCA |
| 19-20 | Build a final capstone project |
| 21+ | Revisit and document all projects on GitHub |

---

## ✅ Final Notes

- KNN is a great starting algorithm for classification tasks.
- It’s simple, powerful, and easy to understand, but not always scalable.
- For large datasets, consider alternatives like Decision Trees or SVMs.
- Keep experimenting with different datasets and hyperparameters.

---

> 📌 “The best way to learn is by doing. Keep practicing and keep building!”
