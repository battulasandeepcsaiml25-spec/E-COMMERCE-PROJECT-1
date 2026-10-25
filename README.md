
# 🛒 Online Purchase Conversion Advisor
### An AI & Machine Learning-Based E-Commerce Purchase Prediction System

## 📌 Project Overview

**Online Purchase Conversion Advisor** is a machine learning project designed to predict whether an online shopping session is likely to result in a purchase.

Many visitors browse e-commerce websites without completing a purchase. This project analyzes browsing behavior and session-related data to identify patterns associated with purchasing decisions.

The system predicts purchase intention based on historical session data and helps e-commerce businesses make data-driven marketing decisions.

**Unit of Analysis:** An online shopping session, not an identified individual customer.

## 🎯 Project Objectives

- Analyze online shopping session data.
- Perform data preprocessing and cleaning.
- Select relevant features for prediction.
- Train a machine learning classification model.
- Predict whether a shopping session will result in a purchase.
- Evaluate model performance using classification metrics.
- Develop a user-friendly interface for purchase prediction.

## 📊 Dataset Information

**Dataset Name:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

- Original records: 12,330 sessions
- Problem Type: Binary Classification
- Target Variable: `Revenue`
- Positive Class: `Revenue = True` (purchase)
- Negative Class: `Revenue = False` (no purchase)

The dataset contains features such as:

- Administrative page visits
- Administrative page duration
- Informational page visits
- Product-related page visits and duration
- Bounce rates
- Exit rates
- Page values
- Special-day information
- Month and visitor type
- Operating system and browser
- Region and traffic type
- Weekend indicator

## 🔄 Project Workflow

The project follows this machine learning workflow:

```text
Dataset Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Machine Learning Model Selection
        ↓
Model Training
        ↓
Prediction
        ↓
Performance Evaluation
        ↓
Save Trained Model
        ↓
Frontend / User Interface
        ↓
Purchase Conversion Result
```

### 1. Dataset Collection
Load the Online Shoppers Purchasing Intention Dataset and understand its structure, features, and target variable.

### 2. Data Preprocessing
- Inspect the dataset and data types.
- Check for missing values and duplicate records.
- Handle data quality issues where necessary.
- Prepare categorical and numerical features.
- Separate input features from the target variable.

### 3. Exploratory Data Analysis (EDA)
Analyze the dataset to understand purchase patterns, feature distributions, class imbalance, and relationships between session behavior and purchase outcomes.

### 4. Feature Selection
Select relevant input features that may help predict purchasing behavior. The `Revenue` column is used as the target and must not be included as an input feature.

### 5. Train-Test Split
Split the dataset into training and testing sets. The training set is used to train the model, while the testing set evaluates its performance on unseen data.

### 6. Machine Learning Model Selection
Choose a suitable classification algorithm for predicting purchase intention. A simple baseline can also be used for comparison.

### 7. Model Training
Train the selected model using the training dataset. Apply the same preprocessing pipeline consistently during training and prediction.

### 8. Prediction
Provide session-related input values to the trained model. The model predicts whether the session is likely to result in a purchase.

### 9. Performance Evaluation
Evaluate the model using appropriate metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC, when appropriate

### 10. Model Saving
Save the trained model and any required preprocessing objects so they can be loaded by the application without retraining every time.

### 11. Frontend Development
Develop a user-friendly interface using HTML, CSS, and JavaScript, or Streamlit, depending on the application implementation.

### 12. Final Prediction Result
Display the predicted purchase outcome and, if supported by the model, the estimated purchase probability.

## 🛠️ Technologies Used

- **Python** — Main programming language
- **Pandas** — Data manipulation and preprocessing
- **NumPy** — Numerical computations
- **Scikit-learn** — Machine learning and model evaluation
- **Joblib** — Saving and loading the trained model
- **HTML** — Frontend structure
- **CSS** — Styling and responsive design
- **JavaScript** — Frontend interactions
- **FastAPI** — Backend API integration, if used
- **Streamlit** — Alternative interface, if used

## 📁 Project Structure

```text
Online-Purchase-Conversion-Advisor/
│
├── dataset.csv
├── preprocessing.py
├── feature_selection.py
├── train_model.py
├── evaluate_model.py
├── predict.py
├── purchase_conversion_model.pkl
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   └── main.py
│
├── requirements.txt
└── README.md
```

*Note: Adjust the filenames and folders to match your actual project structure.*

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Online-Purchase-Conversion-Advisor
```

Replace the placeholder with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by your code. For example:

```bash
pip install pandas numpy scikit-learn joblib
```

For a FastAPI backend, install:

```bash
pip install fastapi uvicorn
```

### Step 4: Run Data Preprocessing

```bash
python preprocessing.py
```

### Step 5: Perform Feature Selection

```bash
python feature_selection.py
```

### Step 6: Train the Machine Learning Model

```bash
python train_model.py
```

### Step 7: Evaluate Model Performance

```bash
python evaluate_model.py
```

### Step 8: Run the Application

For a FastAPI backend, if the entry point is `backend/main.py` and it defines `app`:

```bash
uvicorn backend.main:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

For a Streamlit application, if the entry file is `app.py`:

```bash
streamlit run app.py
```

Use the commands that match the actual filenames in your project.

## 📈 Model Evaluation

The trained model is evaluated on a held-out test dataset to measure its ability to predict online purchase intention.

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall prediction correctness |
| Precision | Measures how many predicted purchases are actual purchases |
| Recall | Measures how many actual purchases are identified |
| F1-score | Balances precision and recall |
| Confusion Matrix | Shows correct and incorrect classifications |
| ROC-AUC | Measures class-ranking performance, when applicable |

Actual evaluation results should be added after running the model.

## 💡 Applications

- E-commerce purchase intention analysis
- Customer session behavior analysis
- Marketing campaign prioritization
- Conversion optimization research
- Data-driven business decision support

## ⚠️ Limitations

- Predictions are based on historical session data.
- A prediction does not guarantee that a purchase will occur.
- Imbalanced data may affect the model's ability to identify purchase sessions.
- Performance may vary across different websites and customer populations.
- The system provides decision support rather than guaranteed sales forecasts.

## 🚀 Future Enhancements

- Compare multiple machine learning algorithms.
- Improve prediction performance through hyperparameter tuning.
- Add interactive data visualization dashboards.
- Display important features influencing predictions.
- Deploy the application online.
- Monitor model performance as shopping behavior changes.

## 🎓 Project Details

**Project Title:** Online Purchase Conversion Advisor

**Domain:** Artificial Intelligence and Machine Learning

**Project Type:** Supervised Machine Learning — Binary Classification

**Dataset:** UCI Online Shoppers Purchasing Intention Dataset

**Objective:** Predict whether an online shopping session is likely to result in a purchase.

---

## 📚 References

Sakar, C. O., et al. *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

---

**Online Purchase Conversion Advisor — Turning Browsing Behavior into Data-Driven Insights.**
