# Telco Customer Churn Prediction using an Artificial Neural Network (ANN)

A beginner-friendly deep learning project that predicts whether a telecom customer will **leave (churn)** or **stay**, using a feed-forward neural network built with **PyTorch**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [How to Run](#how-to-run)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

---

## Project Overview

Customer churn means a customer stops using a company's service. Finding likely churners early helps a business act before it loses them. This project builds a binary classifier that outputs `1` (churn) or `0` (stay) for each customer based on their account and service details.

**Goals**
- Practice the complete deep learning workflow on tabular data
- Handle real-world data problems (text stored as numbers, imbalanced classes)
- Evaluate with the right metrics, not accuracy alone

---

## Dataset

- **Name:** Telco Customer Churn
- **Source:** [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** about 7,000 customers, 21 columns
- **Target:** `Churn` (Yes / No), about 26% of customers churn (imbalanced)

**Feature groups**
| Group | Columns |
|---|---|
| Customer info | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Charges | MonthlyCharges, TotalCharges |

---

## Tech Stack

- Python 3
- PyTorch (model and training)
- scikit-learn (preprocessing, splitting, metrics)
- pandas, NumPy (data handling)
- Matplotlib (loss curves)
- Google Colab (environment)

---

## Project Workflow

### 1. Data Cleaning
- Dropped `customerID` (identifier, not a useful feature)
- `TotalCharges` is stored as text because 11 rows contain a blank space. Converted with `pd.to_numeric(errors="coerce")` and filled the blanks with `0` (these are new customers with `tenure = 0`)

### 2. Encoding
- Applied `LabelEncoder` to all categorical (text) columns, including the target `Churn`
- Verified no text columns or null values remained

### 3. Train / Validation / Test Split
- 60% train, 20% validation, 20% test
- Used `stratify` so the churn ratio is the same in every split

### 4. Feature Scaling
- `StandardScaler` fitted on the **training data only**, then applied to validation and test sets (avoids data leakage)

### 5. Model Training
- Loss: `BCEWithLogitsLoss` (with `pos_weight` to handle class imbalance)
- Optimizer: Adam (`lr=0.001`, `weight_decay=1e-4`)
- Batch size: 32, up to 100 epochs
- **Early stopping** and **best-model checkpointing** based on validation loss

### 6. Evaluation
- Accuracy, confusion matrix, precision, recall and F1-score
- Decision threshold chosen on the validation set (not the test set)
- Training vs validation loss curve to check overfitting

---

## Model Architecture

```
Input (19 features)
   -> Linear(19, 16) -> ReLU -> Dropout(0.3)
   -> Linear(16, 8)  -> ReLU
   -> Linear(8, 1)   -> Sigmoid (applied at prediction time)
```

The output is a single churn probability. The sigmoid is not part of the model because `BCEWithLogitsLoss` applies it internally during training.

---

## Results

| Experiment | Test Accuracy | Churn Recall | Churn F1 |
|---|---|---|---|
| Baseline ANN (32-32, no dropout, last epoch) | 0.792 | 0.53 | 0.57 |
| + pos_weight, dropout, early stopping | _add result_ | _add result_ | _add result_ |
| + threshold tuning | _add result_ | _add result_ | _add result_ |
| Logistic Regression (comparison) | _add result_ | _add result_ | _add result_ |

**Baseline confusion matrix (test set, 1409 customers)**

|  | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actual: Stay** | 920 | 116 |
| **Actual: Churn** | 177 | 196 |

> A model that always predicts "No churn" would already score about 73.5% accuracy. This is why churn **recall** and **F1** matter more than accuracy here.

---

## How to Run

1. Download the dataset from Kaggle and keep the CSV file.
2. Open the notebook in Google Colab (or Jupyter).
3. Install the requirements if running locally:
   ```bash
   pip install torch scikit-learn pandas numpy matplotlib
   ```
4. Upload the CSV (or update the file path in the notebook).
5. Run all cells from top to bottom (**Runtime -> Run all**).

> Run the encoding cell only once per session. Running it twice on already-encoded columns will cause errors.

---

## Key Learnings

- Real datasets have hidden problems (such as `TotalCharges` stored as text) that must be fixed before modelling
- Churn is a **classification** problem, so use BCE loss and classification metrics, not MSE and R²
- Save the **best** epoch, not the last one, to avoid overfitting
- Keep a separate validation set so the test set stays untouched until the end
- On imbalanced data, accuracy can be misleading, so check recall and F1 for the minority class
- A small network with dropout often matches a larger one on small tabular data

---

## Future Improvements

- One-hot encode `PaymentMethod` and `InternetService` instead of label encoding
- Compare against Logistic Regression and Random Forest
- Hyperparameter tuning (layers, neurons, dropout, learning rate)
- Try SMOTE or other resampling methods
- Build a small Streamlit or Flask app where a user enters customer details and gets a churn prediction

---

## Author

**<Your Name>**
Beginner AI/ML learner. Feel free to open an issue or suggest improvements.
