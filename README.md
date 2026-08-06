🏦 Two-Stage Loan Approval & Risk Assessment ML System
Python Scikit-Learn Streamlit License

An end-to-end Machine Learning application that automates credit risk evaluation and dynamic loan amount estimation using a Cascaded Two-Stage ML Architecture.

The system provides real-time predictions via an interactive Streamlit Web Application as well as a command-line interface (CLI).

🎯 Key Features
Cascaded Two-Stage ML Architecture: Combines binary classification for approval with conditional regression for loan limit estimation.
High Predictive Accuracy:
Stage 1 (Classification): Achieves 99.0% Accuracy, 0.99 F1-Score, and 98.42% OOB Score on 4,269 credit profiles.
Stage 2 (Regression): Achieves an 86.44% R² Score for eligible loan limit estimation.
Dual User Interfaces:
Streamlit Web UI: User-friendly web form with instant approval status and amount recommendations.
CLI Application: Fast terminal interface for quick batch testing or script integration.
Modular Production Codebase: Object-oriented pipeline architecture with decoupled data validation, model loading, dynamic inference, and YAML configuration.
⚙️ System Architecture & Workflow
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
Input Features: no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value.
Stage 1 (Approval Classifier): Predicts whether the loan application is Approved or Rejected.
Stage 2 (Amount Regressor): If Approved, Stage 2 triggers automatically to calculate the recommended maximum eligible loan amount.
📊 Model Evaluation & Metrics
Stage	Task	Model	Key Metrics
Stage 1	Approval Classification	RandomForestClassifier (n_estimators=400, 5-fold CV)	Accuracy: 99.0%
F1-Score: 0.99
OOB Score: 98.42%
Stage 2	Eligible Amount Regression	RandomForestRegressor (n_estimators=200, max_depth=8)	R² Score: 86.44%
Top Feature: Annual Income (93.35% Importance)
📁 Repository Structure
.
├── app/
│   ├── __init__.py
│   ├── loader.py        # Model loading utilities (Joblib)
│   ├── predict.py       # Two-stage cascaded prediction logic
│   └── utils.py         # Data transformation & input validation helpers
├── archive/
│   ├── analyze.ipynb    # Complete EDA, feature engineering & model training notebook
│   └── loan_approval_dataset.csv
├── models/              # Pretrained Scikit-Learn pipelines (.pkl)
│   ├── stage_1_rf_classifier_pipeline.pkl
│   └── stage_2_rf_regression_pipeline.pkl
├── config.yaml          # Model paths and default UI configuration
├── streamlit_app.py     # Streamlit Web Application entry point
├── main.py              # Command-Line Interface (CLI) entry point
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
🚀 Quickstart Guide
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/loan-approval-ml-system.git
cd loan-approval-ml-system
2. Set Up Virtual Environment & Install Dependencies
Using uv (Recommended):

uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
Using standard pip:

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
3. Run the Streamlit Web Application
streamlit run streamlit_app.py
Open http://localhost:8501 in your browser.

4. Run the CLI Application
python main.py
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
