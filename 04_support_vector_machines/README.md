# Mastering Support Vector Machines (SVM) – From Basics to Advanced

Welcome to your complete SVM Learning Repository. This repo is designed for those who want to master Support Vector Machines in theory, coding, mathematics, and real-world projects.

---

## What You Will Learn

- Fundamental Concepts of SVM
- Hard Margin vs Soft Margin
- Kernel Trick (Linear, Polynomial, RBF, Sigmoid)
- Mathematical Foundations (explained in simple terms)
- Hyperparameter Tuning
- Real-World Projects from Basic to Advanced
- Model Deployment using FastAPI or Flask

---

## 1. What is Support Vector Machine?

SVM is a supervised machine learning algorithm used for classification and regression tasks. Its main goal is to find the best boundary (called the hyperplane) that separates different classes of data with the maximum margin.

---

## 2. Mathematical Intuition (Simple Form)

### Hard Margin SVM:
- Used when data is perfectly separable.
- The algorithm tries to find the widest possible gap (margin) between classes.
- The goal is to minimize the weight vector while keeping all points correctly classified.

Objective:
- Minimize: 1/2 * ||w||^2
- Subject to: y * (w^T * x + b) >= 1 for all samples

### Soft Margin SVM:
- Used when data is noisy or not linearly separable.
- Allows some misclassifications using slack variables.
- Introduces a penalty parameter C to balance between margin size and error.

Objective:
- Minimize: 1/2 * ||w||^2 + C * sum of slack variables
- Subject to: y * (w^T * x + b) >= 1 - slack for all samples

---

## 3. Hard Margin vs Soft Margin

| Feature           | Hard Margin                   | Soft Margin                        |
|------------------|-------------------------------|------------------------------------|
| Data Requirement | Perfectly linearly separable  | Allows overlapping classes         |
| Tolerance        | No tolerance for error        | Allows some misclassification      |
| Use Case         | Clean datasets                | Noisy or real-world datasets       |

---

## 4. The Kernel Trick

The kernel trick is used to handle non-linearly separable data by mapping it into a higher-dimensional space without explicitly doing the transformation.

### Common Kernels:

- **Linear Kernel**: Works well with linearly separable data.
- **Polynomial Kernel**: Good for curved decision boundaries.
- **RBF (Radial Basis Function)**: Most commonly used; good for complex patterns.
- **Sigmoid Kernel**: Similar to a neural network's activation function.

---

## 5. Hyperparameters in SVM

- **C (Regularization parameter)**: Controls trade-off between a smooth decision boundary and correctly classifying training points.
- **Gamma**: Defines how far the influence of a single point reaches (for RBF, Polynomial, Sigmoid kernels).
- **Degree**: The degree of the polynomial kernel.

Tuning these values is critical for good performance.

---

## 6. Tools & Libraries Used

- Python 3
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Jupyter Notebook
- FastAPI (optional deployment)

---

## 7. Practice Notebooks

| File                                | Description                                    |
|-------------------------------------|------------------------------------------------|
| 01_svm_linearly_separable.ipynb     | Hard margin SVM on simple dataset              |
| 02_soft_margin_svm.ipynb            | Soft margin with noisy data                   |
| 03_kernel_trick.ipynb               | Comparison of linear, polynomial, RBF kernels |
| 04_hyperparameter_tuning.ipynb      | Using GridSearchCV to tune C and gamma        |
| 05_svm_vs_logistic.ipynb            | Compare SVM with logistic regression          |

---

## 8. Project Roadmap

### Basic Project: Iris Flower Classification
- Dataset: Iris (from scikit-learn)
- Classify Setosa vs Versicolor
- Uses linear SVM
- Includes visualization

### Medium Project 1: MNIST Digit Classification
- Dataset: MNIST digits (0–9)
- Use RBF kernel
- Includes PCA for dimensionality reduction

### Medium Project 2: Breast Cancer Detection
- Dataset: Wisconsin Breast Cancer
- Binary classification: Benign vs Malignant
- Includes hyperparameter tuning and metrics

### Advanced Project 1: Email Spam Detection
- Dataset: UCI Spambase
- Text preprocessing and feature extraction (TF-IDF)
- Use SVM with RBF kernel
- Evaluate precision, recall, F1-score
- Optional: Deploy with FastAPI

### Advanced Project 2: Sentiment Analysis
- Dataset: IMDb reviews or Twitter data
- Text cleaning and vectorization
- Linear and RBF SVM models
- Bonus: Deploy using Flask/FastAPI

---

## 9. Folder Structure

