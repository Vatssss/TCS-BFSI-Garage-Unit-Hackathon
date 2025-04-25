# 🏦 German Credit Risk Prediction

A machine learning application to predict the **creditworthiness of loan applicants** using the German Credit dataset. Built for the **TCS BFSI Garage Hackathon**, this solution combines data preprocessing, ensemble modeling, and a user-friendly Streamlit interface for real-world usability.

---

## 🚀 Demo

🎥 [Watch the demo video with voiceover →](#) *(Insert your YouTube or drive link here)*\
📊 Live app: [Live link](https://german-credit-risk-predictor.streamlit.app/)

---

## 📂 Project Structure

```
.
🔍— app.py                      # Streamlit front-end application
🔍— generate_feature_importance.py  # Script to generate and save feature importances
🔍— voting_model.pkl           # Trained ensemble model
🔍— scaler.pkl                 # StandardScaler object for input features
🔍— feature_names.pkl          # Ordered list of feature names
🔍— feature_importance.csv     # Output of top features (generated)
🔍— german_credit_data.csv     # Original dataset
🔍— requirements.txt           # Python dependencies
🔍— README.md                  # Project documentation
```

---

## 🧐 Problem Statement

> **Objective:** Predict whether a loan applicant is a **Good** or **Bad** credit risk using historical banking data.

### Features:

- **Demographic:** Age, Sex, Job
- **Financial Behavior:** Housing, Saving Account, Checking Account
- **Loan Details:** Credit Amount, Duration, Purpose
- **Target:** `Risk` (good = 0, bad = 1)

---

## 🔍 Data Exploration & Preprocessing

### ✅ Key Preprocessing Steps:

- **Missing Values Handling:** Imputed as `"no_info"` to retain signal from absence of accounts.
- **Categorical Encoding:**
  - Binary encoding for `Sex`, `Risk`
  - One-hot encoding for `Housing`, `Saving accounts`, `Checking account`, `Purpose`
- **Feature Engineering:**
  - Age binned into: `Young`, `Adult`, `Senior`, `Elder`
  - Credit amount binned into: `Low`, `Medium`, `High`, `Very High`
- **Scaling:** StandardScaler applied to numeric inputs

---

## 🤖 Model Development

Trained and evaluated multiple models:

| Model                         | Accuracy | ROC-AUC   | F1 Score (Bad Risk) |
| ----------------------------- | -------- | --------- | ------------------- |
| Logistic Regression           | 0.68     | 0.76      | 0.57                |
| Random Forest                 | 0.77     | 0.77      | 0.48                |
| XGBoost                       | 0.76     | 0.76      | 0.59                |
| **Voting Classifier (Final)** | **0.77** | **0.775** | **0.59**            |

Final model = **Soft Voting Ensemble** of the three classifiers above.

---

## 📊 Feature Importance

Run `generate_feature_importance.py` to compute and export feature importances:

```bash
python generate_feature_importance.py
```

Top predictive features include:

- Duration
- Credit Amount
- Checking Account
- Purpose of Loan

---

## 💻 Streamlit Web App

Launch the app locally:

```bash
streamlit run app.py
```

### Features:

- Interactive form for user input
- Real-time prediction with ensemble model
- Risk probability display + progress bar
- Feature importance bar chart

---

## ⚙️ Setup Instructions

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/german-credit-risk.git
cd german-credit-risk
```

### 2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

Key libraries:

- scikit-learn
- xgboost
- pandas
- numpy
- streamlit
- joblib

---

## 👤 Author

**Vatsal Saxena**\
Built for the **TCS BFSI Garage Hackathon**

