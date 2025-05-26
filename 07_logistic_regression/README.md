# 🔍 Logistic Regression: Theory & Projects (From Basic to Advanced)

This repository is a complete guide to mastering **Logistic Regression** — a powerful and widely used classification algorithm. It covers all key concepts in simple language and provides projects for hands-on practice, progressing from basic to advanced levels.

---

## 📐 Mathematical Concepts (Explained Simply)

### 1. Sigmoid Function

- The sigmoid function is used to convert any real number into a value between 0 and 1.
- This output represents the **probability** that the input belongs to class 1 (positive class).
- Formula: 1 / (1 + e^(-z)), where z = weighted sum of input features.
- Example: If z = 2, sigmoid(2) ≈ 0.88 → meaning 88% chance it’s class 1.

---

### 2. Decision Boundary

- After calculating the probability using the sigmoid function, we need to make a decision.
- We typically use a threshold of 0.5:
  - If the probability is **greater than or equal to 0.5**, predict **class 1**
  - If the probability is **less than 0.5**, predict **class 0**
- The line (or surface) that separates these two classes is called the **decision boundary**.

---

### 3. Loss Function (Binary Cross Entropy)

- The loss function measures how far the model's predictions are from the actual values.
- For logistic regression, we use **Binary Cross Entropy** loss.
- If the actual class is 1 and the predicted probability is close to 1, loss is small (good).
- If the actual class is 0 and the predicted probability is close to 1, loss is high (bad).
- The model tries to **minimize this loss** during training.

---

### 4. Gradient Descent (Learning the Best Parameters)

- Gradient Descent is the algorithm used to **update the weights and bias** so that the loss becomes smaller.
- It works by calculating the direction in which the loss increases the most, then **moving in the opposite direction** (i.e., minimizing the loss).
- The size of the step taken during each update is controlled by the **learning rate**.

---

### 5. Evaluation Metrics

Once the model is trained, we evaluate how well it performs using:
- **Accuracy**: What percentage of predictions were correct?
- **Precision**: Out of all predicted positives, how many were actually positive?
- **Recall**: Out of all actual positives, how many did we correctly predict?
- **F1 Score**: A balance between precision and recall.
- **ROC-AUC**: A score that tells how well the model separates the two classes.

---

## 🧪 Projects List

### 🔹 Basic Projects
1. **Iris Binary Classifier** – Classify if a flower is Iris-Versicolor or not  
2. **Student Admission Prediction** – Predict if a student gets admitted based on test scores  
3. **Spam Email Detection** – Detect whether an email is spam or not  
4. **Tumor Diagnosis** – Predict if a tumor is malignant (harmful) or benign (safe)  

---

### 🔸 Medium Projects
1. **Titanic Survival Prediction** – Predict if a passenger survived the Titanic crash  
2. **Customer Churn Prediction** – Predict if a customer will leave a telecom company  
3. **Bank Loan Default Prediction** – Predict if a person will fail to repay a loan  
4. **Breast Cancer Diagnosis** – Predict type of tumor using real medical data  

---

### 🔺 Advanced Projects
1. **Credit Card Fraud Detection** – Detect fraudulent transactions (highly imbalanced data)  
2. **Fake News Detection** – Predict if a news article is real or fake (text classification)  
3. **Sentiment Analysis** – Predict if a review is positive or negative (NLP)  
4. **HR Attrition Analysis** – Predict if an employee will leave the company  
5. **MNIST Handwritten Digits** – Use Logistic Regression to classify digits (0–9) using softmax for multiclass classification  

---

This repository helps you **understand the theory** and **build practical skills** with real-world datasets. No black boxes — everything is explained and implemented from scratch and with libraries.
