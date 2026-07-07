# Customer Action Prediction using K-Nearest Neighbors (K-NN) 🤖

This repository contains a classic supervised machine learning pipeline implementing the **K-Nearest Neighbors (K-NN)** algorithm. The model analyzes user demographics (Age and Estimated Salary) to predict binary consumer behavior outcomes with a high degree of spatial precision.

## 🧠 Model Configuration
Unlike parametric models that calculate explicit line coefficients, K-NN is an instance-based lazy learner. It classifies data points dynamically based on their closest neighbors:
* **Neighbors (\(K\))**: `n_neighbors=5` (Optimized to balance bias and variance).
* **Distance Metric**: `metric='minkowski'` with `p=2` (Equivalent to standard Euclidean distance).

## 🛠️ Pipeline Architecture

### 1. Mandatory Feature Scaling
Because K-NN computes geometric distances between coordinates, columns with larger numeric scales (like \$87,000 salary) will completely overwhelm smaller scales (like 30 years of age). To resolve this, a `StandardScaler` was applied to normalize all features to a uniform variance scale before fitting.

```python
# Real-time inference wrapper using scale transformation
classifier.predict(sc.transform([[30, 87000]]))
```

### 2. Model Evaluation Breakdown
The trained classifier was evaluated against an un-seen validation testing matrix, yielding a final **Accuracy Score of 93.0%**.

#### Confusion Matrix Results:
```text
[[64  4]   -->  [True Negatives (64),  False Positives (4)]
 [ 3 29]]  -->  [False Negatives (3),  True Positives (29)]
```
* **True Negatives**: 64 instances correctly flagged as `0`.
* **True Positives**: 29 instances correctly flagged as `1`.
* **Total Misclassifications**: Only 7 instances out of 100 samples.

## 💻 Tech Stack
* **Language**: Python 3
* **Environment**: Jupyter Notebook / Google Colab
* **Libraries**: `scikit-learn` (Inference & Evaluation), `matplotlib` / `ListedColormap` (Boundary Visualization), `pandas`, `numpy`
