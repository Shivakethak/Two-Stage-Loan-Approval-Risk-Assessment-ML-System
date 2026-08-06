# 🏦 Two-Stage Loan Approval & Risk Assessment ML System

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning application that automates credit risk evaluation and dynamic loan amount estimation using a **Cascaded Two-Stage ML Architecture**. 

The system provides real-time predictions via an interactive **Streamlit Web Application** as well as a command-line interface (**CLI**).

---

## 🎯 Key Features

- **Cascaded Two-Stage ML Architecture**: Combines binary classification for approval with conditional regression for loan limit estimation.
- **High Predictive Accuracy**:
  - **Stage 1 (Classification)**: Achieves **99.0% Accuracy**, **0.99 F1-Score**, and **98.42% OOB Score** on 4,269 credit profiles.
  - **Stage 2 (Regression)**: Achieves an **86.44% R² Score** for eligible loan limit estimation.
- **Dual User Interfaces**:
  - **Streamlit Web UI**: User-friendly web form with instant approval status and amount recommendations.
  - **CLI Application**: Fast terminal interface for quick batch testing or script integration.
- **Modular Production Codebase**: Object-oriented pipeline architecture with decoupled data validation, model loading, dynamic inference, and YAML configuration.

---

## ⚙️ System Architecture & Workflow

```
             +----------------------------+
             |    Applicant Data Input    |
             +----------------------------+
                           |
                           v
             +----------------------------+
             | Preprocessing Pipeline     |
             | (Imputer, Scaler, OneHot)  |
             +----------------------------+
                           |
                           v
             +----------------------------+
             | STAGE 1: Classifier        |
             | (RandomForestClassifier)   |
             +----------------------------+
                    /              \
           Approved/                \Rejected
                  v                  v
    +----------------------------+   +-------------------+
    | STAGE 2: Regressor         |   | ❌ Loan Rejected  |
    | (RandomForestRegressor)    |   +-------------------+
    +----------------------------+
                  |
                  v
    +----------------------------+
    | ✅ Eligible Amount Output  |
    +----------------------------+
```

---

## 📊 Model Evaluation & Metrics Table

| Stage | Task | Model | Key Metrics |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Approval Classification | `RandomForestClassifier` (n_estimators=400, 5-fold CV) | **Accuracy: 99.0%**<br>**F1-Score: 0.99**<br>**OOB Score: 98.42%** |
| **Stage 2** | Eligible Amount Regression | `RandomForestRegressor` (n_estimators=200, max_depth=8) | **R² Score: 86.44%**<br>Top Feature: Annual Income (93.35% Importance) |

---

## 💻 Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.12** | Primary programming language |
| **Scikit-Learn** | Machine Learning pipelines, `RandomForest`, `ColumnTransformer`, `GridSearchCV` |
| **Streamlit** | Interactive Web GUI & real-time prediction dashboard |
| **Pandas & NumPy** | Data manipulation and preprocessing |
| **Joblib** | Serialization and loading of pre-trained pipelines |
| **PyYAML** | Externalized configuration management (`config.yaml`) |
| **UV** | Fast Python virtual environment and dependency manager |

---

## 📁 Repository Structure

```
.
├── code/
│   └── Loan_approval_final/
│       ├── app/
│       │   ├── loader.py        # Model loading utilities (Joblib)
│       │   ├── predict.py       # Two-stage cascaded prediction logic
│       │   └── utils.py         # Data transformation & input validation helpers
│       ├── archive/
│       │   ├── analyze.ipynb    # Complete EDA, feature engineering & model training notebook
│       │   └── loan_approval_dataset.csv
│       ├── models/              # Pretrained Scikit-Learn pipelines (.pkl)
│       │   ├── stage_1_rf_classifier_pipeline.pkl
│       │   └── stage_2_rf_regression_pipeline.pkl
│       ├── config.yaml          # Model paths and default UI configuration
│       ├── streamlit_app.py     # Streamlit Web Application entry point
│       └── main.py              # Command-Line Interface (CLI) entry point
├── dataset/
├── document/
├── Evaluation/
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-ml-system.git
cd "1. Loan Approval ML System (Final)"
```

### 2. Set Up Virtual Environment & Install Dependencies

Using `uv` (Recommended):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r code/Loan_approval_final/requirements.txt
```

Using standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r code/Loan_approval_final/requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
cd code/Loan_approval_final
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run the CLI Application
```bash
cd code/Loan_approval_final
python main.py
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
