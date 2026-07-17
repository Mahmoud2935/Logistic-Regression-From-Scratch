# Logistic Regression From Scratch

A complete implementation of Logistic Regression built entirely from scratch using Python without relying on machine learning libraries such as Scikit-Learn.

The purpose of this project was to gain a deeper understanding of how Logistic Regression works internally by implementing the core components manually, including gradient descent, feature scaling, train/test splitting, model evaluation, and visualization.

Rather than focusing solely on model accuracy, this project emphasizes understanding the complete machine learning workflow and the mathematical intuition behind the algorithm.

---

## Dataset

This project uses a student performance dataset for binary classification.

### Target

- Pass (1)
- Fail (0)

### Example Features

- Hours Studied
- Sleep Hours
- Previous Scores
- Attendance

---

## Project Goal

The main goal of this project was to understand how Logistic Regression works under the hood by implementing every major component manually instead of relying on machine learning libraries.

This project focuses on learning the training process, optimization techniques, preprocessing pipeline, model evaluation, and common machine learning practices such as preventing data leakage.

---

## Features

- Logistic Regression from scratch
- Multi-feature support
- Gradient Descent optimization
- Sigmoid activation function
- Binary classification
- Min-Max Normalization
- Train/Test Split
- Data Leakage prevention
- Early Stopping
- Accuracy evaluation
- Confusion Matrix evaluation
- Visualization Dashboard

---

## Project Structure

```text
Logistic-Regression-From-Scratch/

├── data/
│   └── student_pass.csv
│
├── Models/
│   └── logistic_regression.py
│
├── preprocessing/
│   └── preprocessing.py
│
├── Metrics/
│   └── metrics.py
│
├── Notes/
│   └── lessons_learned.md
│
├── experiments/
│   ├── visualization.py
│   └── losses.json
│
├── dashboard.png
├── README.md
└── train.py
```

---

## Implemented From Scratch

The following components were implemented manually without using machine learning libraries:

- Sigmoid Function
- Logistic Regression
- Gradient Descent
- Weight Updates
- Bias Updates
- Train/Test Splitting
- Min-Max Normalization
- Classification Threshold
- Accuracy Metric
- Confusion Matrix
- Early Stopping

---

## Machine Learning Pipeline

```text
Load Dataset
      ↓
Shuffle Data
      ↓
Train/Test Split
      ↓
Normalize Train Data
      ↓
Normalize Test Data
      ↓
Train Logistic Regression
      ↓
Generate Probabilities
      ↓
Classify Predictions
      ↓
Evaluate Accuracy
      ↓
Analyze Results
      ↓
Visualization Dashboard
```

---

## Visualization Dashboard

The project includes a dashboard containing:

- Pass vs Fail Distribution
- Correlation Matrix
- Hours Distribution by Pass Status
- Pass Rate vs Hours
- Training Loss Over Epochs

---

## Dashboard Preview

![Dashboard](dashboard.png)

---

## Example Results

| Metric | Value |
|----------|----------|
| Train Size | 168 |
| Test Size | 42 |
| Accuracy | 95.24% |
| TN | 23 |
| FP | 1 |
| FN | 1 |
| TP | 17 |

The model correctly classified 40 out of 42 test samples.

---

## Challenges Faced

During this project, I faced several challenges:

- Understanding how gradient descent updates weights and bias.
- Preventing data leakage during preprocessing.
- Choosing a suitable learning rate.
- Implementing train/test splitting manually.
- Adding early stopping without machine learning libraries.
- Building evaluation metrics from scratch.

---

## Concepts Learned

During this project, I learned:

- Samples vs Features
- Weights and Bias
- Logistic Regression
- Gradient Descent
- Feature Scaling
- Normalization
- Train/Test Split
- Data Leakage
- Model Evaluation
- Visualization and Analysis
- Early Stopping
- Confusion Matrix

---

## Key Takeaways

- Logistic Regression predicts probabilities rather than direct class labels.
- Feature scaling significantly improves gradient descent performance.
- Data leakage can lead to misleading evaluation results.
- Training and testing data should always be processed separately.
- Early stopping can reduce unnecessary training and improve efficiency.
- Understanding algorithms from scratch provides deeper intuition than simply using libraries.
- Building algorithms manually improves debugging skills and machine learning intuition.

---

## Future Improvements

- Add Precision, Recall, and F1 Score.
- Add ROC Curve visualization.
- Support multiclass classification.
- Compare results with a Scikit-Learn implementation.
- Save and load trained models.

---

## Run

```bash
python train.py
```

---

## Author

**Mahmoud Mostfa Taha**

Machine Learning Student
