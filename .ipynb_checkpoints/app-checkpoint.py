from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime
import json
import numpy as np

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# LOANIQ FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = "models/loanIQ_logistic_model.pkl"

FINAL_THRESHOLD = 0.45

DATA_PATH = "data/Loan.csv"


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# HIDDEN FEATURE DEFAULTS
# =========================================================
# Users do not need to enter these fields manually.
# =========================================================

DEFAULTS = {

    "CreditCardUtilizationRate": 0.266116409,

    "NumberOfOpenCreditLines": 3,

    "NumberOfCreditInquiries": 1,

    "DebtToIncomeRatio": 0.263977373,

    "PaymentHistory": 24,

    "CheckingAccountBalance": 1116,

    "TotalLiabilities": 22174.5,

    "UtilityBillsPaymentHistory": 0.820690646,

    "JobTenure": 5,

}


# =========================================================
# DECISION EXPLANATION
# =========================================================

def get_decision_explanation(input_df, model, top_n=5):

    """
    Generate a model-based explanation for
    an individual Logistic Regression prediction.
    """

    try:

        # -------------------------------------------------
        # Get preprocessing and classifier
        # -------------------------------------------------

        preprocessor = model.named_steps["preprocessor"]

        classifier = model.named_steps["classifier"]


        # -------------------------------------------------
        # Transform applicant data
        # -------------------------------------------------

        transformed_data = preprocessor.transform(
            input_df
        )


        # -------------------------------------------------
        # Get feature names
        # -------------------------------------------------

        feature_names = (
            preprocessor.get_feature_names_out()
        )


        # -------------------------------------------------
        # Get Logistic Regression coefficients
        # -------------------------------------------------

        coefficients = classifier.coef_[0]


        # -------------------------------------------------
        # Calculate feature contributions
        # -------------------------------------------------

        contributions = (
            transformed_data[0] * coefficients
        )


        # -------------------------------------------------
        # Convert sparse result if necessary
        # -------------------------------------------------

        if hasattr(contributions, "toarray"):

            contributions = (
                contributions
                .toarray()
                .ravel()
            )

        else:

            contributions = np.asarray(
                contributions
            ).ravel()


        # -------------------------------------------------
        # Create explanation dataframe
        # -------------------------------------------------

        explanation_df = pd.DataFrame({

            "Feature": feature_names,

            "Contribution": contributions

        })


        # -------------------------------------------------
        # Absolute contribution
        # -------------------------------------------------

        explanation_df["AbsoluteContribution"] = (
            explanation_df["Contribution"].abs()
        )


        # -------------------------------------------------
        # Sort by strongest influence
        # -------------------------------------------------

        explanation_df = (
            explanation_df
            .sort_values(
                "AbsoluteContribution",
                ascending=False
            )
        )


        # -------------------------------------------------
        # Positive factors
        # -------------------------------------------------

        positive = explanation_df[
            explanation_df["Contribution"] > 0
        ].head(top_n)


        # -------------------------------------------------
        # Negative factors
        # -------------------------------------------------

        negative = explanation_df[
            explanation_df["Contribution"] < 0
        ].head(top_n)


        # -------------------------------------------------
        # Clean feature names
        # -------------------------------------------------

        def clean_feature_name(name):

            name = name.replace(
                "num__",
                ""
            )

            name = name.replace(
                "cat__",
                ""
            )

            name = name.replace(
                "_",
                " "
            )

            return name


        # -------------------------------------------------
        # Build positive factors
        # -------------------------------------------------

        positive_factors = []

        for _, row in positive.iterrows():

            positive_factors.append({

                "feature":
                    clean_feature_name(
                        row["Feature"]
                    ),

                "contribution":
                    round(
                        float(
                            row["Contribution"]
                        ),
                        3
                    )

            })


        # -------------------------------------------------
        # Build negative factors
        # -------------------------------------------------

        negative_factors = []

        for _, row in negative.iterrows():

            negative_factors.append({

                "feature":
                    clean_feature_name(
                        row["Feature"]
                    ),

                "contribution":
                    round(
                        float(
                            row["Contribution"]
                        ),
                        3
                    )

            })


        return (
            positive_factors,
            negative_factors
        )


    except Exception as e:

        print(
            "Explanation error:",
            e
        )

        return [], []


# =========================================================
# LANDING PAGE / DASHBOARD
# =========================================================

@app.route("/")
def home():

    return render_template(
        "dashboard.html"
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

@app.route("/predict", methods=["GET"])
def predict_page():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION PROCESS
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =================================================
        # APPLICATION DATE
        # =================================================

        application_date = request.form[
            "ApplicationDate"
        ]


        date_obj = datetime.strptime(
            application_date,
            "%Y-%m-%d"
        )


        application_year = date_obj.year

        application_month = date_obj.month

        application_day_of_week = (
            date_obj.weekday()
        )


        # =================================================
        # USER INPUT
        # =================================================

        input_data = {

            # -------------------------------------------------
            # Personal Information
            # -------------------------------------------------

            "Age":
                int(
                    request.form["Age"]
                ),

            "AnnualIncome":
                int(
                    request.form["AnnualIncome"]
                ),

            "CreditScore":
                int(
                    request.form["CreditScore"]
                ),

            "EmploymentStatus":
                request.form[
                    "EmploymentStatus"
                ],

            "EducationLevel":
                request.form[
                    "EducationLevel"
                ],

            "Experience":
                int(
                    request.form["Experience"]
                ),

            "LoanAmount":
                int(
                    request.form["LoanAmount"]
                ),

            "LoanDuration":
                int(
                    request.form["LoanDuration"]
                ),

            "MaritalStatus":
                request.form[
                    "MaritalStatus"
                ],

            "NumberOfDependents":
                int(
                    request.form[
                        "NumberOfDependents"
                    ]
                ),

            "HomeOwnershipStatus":
                request.form[
                    "HomeOwnershipStatus"
                ],


            # -------------------------------------------------
            # Financial Information
            # -------------------------------------------------

            "MonthlyDebtPayments":
                int(
                    request.form[
                        "MonthlyDebtPayments"
                    ]
                ),

            "SavingsAccountBalance":
                int(
                    request.form[
                        "SavingsAccountBalance"
                    ]
                ),

            "NetWorth":
                int(
                    request.form["NetWorth"]
                ),


            # -------------------------------------------------
            # Credit Information
            # -------------------------------------------------

            "BankruptcyHistory":
                int(
                    request.form[
                        "BankruptcyHistory"
                    ]
                ),

            "PreviousLoanDefaults":
                int(
                    request.form[
                        "PreviousLoanDefaults"
                    ]
                ),

            "LengthOfCreditHistory":
                int(
                    request.form[
                        "LengthOfCreditHistory"
                    ]
                ),


            # -------------------------------------------------
            # Loan Information
            # -------------------------------------------------

            "LoanPurpose":
                request.form[
                    "LoanPurpose"
                ],


            # -------------------------------------------------
            # Automatically Generated Date Features
            # -------------------------------------------------

            "ApplicationYear":
                application_year,

            "ApplicationMonth":
                application_month,

            "ApplicationDayOfWeek":
                application_day_of_week,


            # -------------------------------------------------
            # Hidden Model Features
            # -------------------------------------------------

            "CreditCardUtilizationRate":
                DEFAULTS[
                    "CreditCardUtilizationRate"
                ],

            "NumberOfOpenCreditLines":
                DEFAULTS[
                    "NumberOfOpenCreditLines"
                ],

            "NumberOfCreditInquiries":
                DEFAULTS[
                    "NumberOfCreditInquiries"
                ],

            "DebtToIncomeRatio":
                DEFAULTS[
                    "DebtToIncomeRatio"
                ],

            "PaymentHistory":
                DEFAULTS[
                    "PaymentHistory"
                ],

            "CheckingAccountBalance":
                DEFAULTS[
                    "CheckingAccountBalance"
                ],

            "TotalLiabilities":
                DEFAULTS[
                    "TotalLiabilities"
                ],

            "UtilityBillsPaymentHistory":
                DEFAULTS[
                    "UtilityBillsPaymentHistory"
                ],

            "JobTenure":
                DEFAULTS[
                    "JobTenure"
                ],

            "TotalDebtToIncomeRatio":
                DEFAULTS[
                    "DebtToIncomeRatio"
                ]

        }


        # =================================================
        # CREATE DATAFRAME
        # =================================================

        input_df = pd.DataFrame(
            [input_data]
        )


        # =================================================
        # MODEL PREDICTION
        # =================================================

        probability = model.predict_proba(input_df)[0][1]


        # =================================================
        # DECISION EXPLANATION
        # =================================================

        positive_factors, negative_factors = (
            get_decision_explanation(
                input_df,
                model
            )
        )


        # =================================================
        # APPLY FINAL THRESHOLD
        # =================================================

        prediction = (
            probability >= FINAL_THRESHOLD
        )


        # =================================================
        # LOAN DECISION
        # =================================================

        if prediction:

            decision = "Loan Approved"

        else:

            decision = "Loan Not Approved"


        # =================================================
        # RISK LEVEL
        # =================================================
        # Application-level indicator based on
        # predicted approval probability.
        # =================================================

        if probability >= 0.75:

            risk_level = "Low Risk"

        elif probability >= FINAL_THRESHOLD:

            risk_level = "Moderate Risk"

        else:

            risk_level = "High Risk"


        # =================================================
        # DISPLAY RESULT
        # =================================================

        return render_template(

            "index.html",

            prediction=decision,

            probability=round(
                probability * 100,
                2
            ),

            risk_level=risk_level,

            positive_factors=positive_factors,

            negative_factors=negative_factors,

            submitted=True

        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return render_template(

            "index.html",

            error=str(e),

            submitted=False

        )


# =========================================================
# ANALYTICS PAGE
# =========================================================

@app.route("/analytics")
def analytics():

    df = pd.read_csv(
        DATA_PATH
    )


    # =================================================
    # BASIC KPIs
    # =================================================

    total_applications = len(df)

    approved = int(
        df["LoanApproved"].sum()
    )

    rejected = (
        total_applications - approved
    )

    approval_rate = round(
        approved /
        total_applications *
        100,
        2
    )

    avg_loan_amount = round(
        df["LoanAmount"].mean(),
        2
    )

    avg_credit_score = round(
        df["CreditScore"].mean(),
        2
    )

    avg_income = round(
        df["AnnualIncome"].mean(),
        2
    )


    # =================================================
    # APPROVAL DISTRIBUTION
    # =================================================

    approval_distribution = {

        "labels": [
            "Approved",
            "Not Approved"
        ],

        "values": [
            approved,
            rejected
        ]

    }


    # =================================================
    # APPROVAL BY EMPLOYMENT
    # =================================================

    employment = (
        df.groupby(
            "EmploymentStatus"
        )["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )


    approval_by_employment = {

        "labels":
            employment.index.tolist(),

        "values":
            employment.values.tolist()

    }


    # =================================================
    # APPROVAL BY EDUCATION
    # =================================================

    education = (
        df.groupby(
            "EducationLevel"
        )["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )


    approval_by_education = {

        "labels":
            education.index.tolist(),

        "values":
            education.values.tolist()

    }


    # =================================================
    # APPROVAL BY HOME OWNERSHIP
    # =================================================

    home = (
        df.groupby(
            "HomeOwnershipStatus"
        )["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )


    approval_by_home = {

        "labels":
            home.index.tolist(),

        "values":
            home.values.tolist()

    }


    # =================================================
    # APPROVAL BY LOAN PURPOSE
    # =================================================

    purpose = (
        df.groupby(
            "LoanPurpose"
        )["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )


    approval_by_purpose = {

        "labels":
            purpose.index.tolist(),

        "values":
            purpose.values.tolist()

    }


    # =================================================
    # CREDIT SCORE DISTRIBUTION
    # =================================================

    credit_score = {

        "approved": df.loc[
            df["LoanApproved"] == 1,
            "CreditScore"
        ].tolist(),

        "rejected": df.loc[
            df["LoanApproved"] == 0,
            "CreditScore"
        ].tolist()

    }


    # =================================================
    # INCOME DISTRIBUTION
    # =================================================

    income = {

        "approved": df.loc[
            df["LoanApproved"] == 1,
            "AnnualIncome"
        ].tolist(),

        "rejected": df.loc[
            df["LoanApproved"] == 0,
            "AnnualIncome"
        ].tolist()

    }


    # =================================================
    # LOAN AMOUNT DISTRIBUTION
    # =================================================

    loan_amount = {

        "approved": df.loc[
            df["LoanApproved"] == 1,
            "LoanAmount"
        ].tolist(),

        "rejected": df.loc[
            df["LoanApproved"] == 0,
            "LoanAmount"
        ].tolist()

    }


    # =================================================
    # DEBT-TO-INCOME RATIO
    # =================================================

    dti = {

        "approved": df.loc[
            df["LoanApproved"] == 1,
            "TotalDebtToIncomeRatio"
        ].tolist(),

        "rejected": df.loc[
            df["LoanApproved"] == 0,
            "TotalDebtToIncomeRatio"
        ].tolist()

    }


    # =================================================
    # PREVIOUS DEFAULTS
    # =================================================

    defaults = (
        df.groupby(
            "PreviousLoanDefaults"
        )["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )


    approval_by_defaults = {

        "labels": [
            "No Previous Default",
            "Previous Default"
        ],

        "values": [

            defaults.get(
                0,
                0
            ),

            defaults.get(
                1,
                0
            )

        ]

    }


    # =================================================
    # BANKRUPTCY
    # =================================================

    bankruptcy = (
        df.groupby(
            "BankruptcyHistory"
        )["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )


    approval_by_bankruptcy = {

        "labels": [
            "No Bankruptcy",
            "Bankruptcy"
        ],

        "values": [

            bankruptcy.get(
                0,
                0
            ),

            bankruptcy.get(
                1,
                0
            )

        ]

    }


    # =================================================
    # RENDER ANALYTICS PAGE
    # =================================================

    return render_template(

        "analytics.html",

        total_applications=
            total_applications,

        approved=
            approved,

        rejected=
            rejected,

        approval_rate=
            approval_rate,

        avg_loan_amount=
            avg_loan_amount,

        avg_credit_score=
            avg_credit_score,

        avg_income=
            avg_income,

        approval_distribution=
            json.dumps(
                approval_distribution
            ),

        approval_by_employment=
            json.dumps(
                approval_by_employment
            ),

        approval_by_education=
            json.dumps(
                approval_by_education
            ),

        approval_by_home=
            json.dumps(
                approval_by_home
            ),

        approval_by_purpose=
            json.dumps(
                approval_by_purpose
            ),

        credit_score=
            json.dumps(
                credit_score
            ),

        income=
            json.dumps(
                income
            ),

        loan_amount=
            json.dumps(
                loan_amount
            ),

        dti=
            json.dumps(
                dti
            ),

        approval_by_defaults=
            json.dumps(
                approval_by_defaults
            ),

        approval_by_bankruptcy=
            json.dumps(
                approval_by_bankruptcy
            )

    )


# =========================================================
# MODEL INSIGHTS PAGE
# =========================================================

@app.route("/model")
def model_insights():
    try:
        # =========================================================
        # DATASET INFORMATION
        # =========================================================

        total_records = 0
        total_features = 0
        approved_count = 0
        not_approved_count = 0
        preview_data = []

        if os.path.exists(DATA_PATH):

            df = pd.read_csv(DATA_PATH)

            total_records = len(df)

            # Original dataset has 36 columns
            total_features = len(df.columns)

            if "LoanApproved" in df.columns:

                approved_count = int(
                    (df["LoanApproved"] == 1).sum()
                )

                not_approved_count = int(
                    (df["LoanApproved"] == 0).sum()
                )

            # -----------------------------------------------------
            # 10 ROW CSV PREVIEW
            # -----------------------------------------------------

            preview_columns = [
                "ApplicationDate",
                "Age",
                "AnnualIncome",
                "CreditScore",
                "LoanAmount",
                "LoanDuration",
                "EmploymentStatus",
                "EducationLevel",
                "LoanPurpose",
                "LoanApproved",
                "RiskScore"
            ]

            preview_columns = [
                col for col in preview_columns
                if col in df.columns
            ]

            preview_df = df[preview_columns].head(10).copy()

            if "ApplicationDate" in preview_df.columns:

                preview_df["ApplicationDate"] = (
                    preview_df["ApplicationDate"]
                    .astype(str)
                    .str[:10]
                )

            preview_data = (
                preview_df
                .fillna("")
                .to_dict(orient="records")
            )

        # =========================================================
        # MODEL PERFORMANCE
        # =========================================================

        metrics = {
            "accuracy": 93.33,
            "precision": 87.57,
            "recall": 84.00,
            "f1": 85.74,
            "roc_auc": 97.86
        }

        # =========================================================
        # CONFUSION MATRIX
        # FINAL LOGISTIC REGRESSION - THRESHOLD 0.45
        # =========================================================

        confusion_matrix = {
            "tn": 2911,
            "fp": 133,
            "fn": 136,
            "tp": 820
        }

        # =========================================================
        # MODEL COMPARISON
        # =========================================================

        model_comparison = [
            {
                "name": "Logistic Regression",
                "accuracy": 93.22,
                "precision": 86.95,
                "recall": 84.31,
                "f1": 85.61,
                "roc_auc": 97.87
            },
            {
                "name": "Tuned Logistic Regression",
                "accuracy": 93.20,
                "precision": 86.93,
                "recall": 84.21,
                "f1": 85.55,
                "roc_auc": 97.87
            },
            {
                "name": "Gradient Boosting",
                "accuracy": 92.75,
                "precision": 87.42,
                "recall": 81.38,
                "f1": 84.29,
                "roc_auc": 97.46
            },
            {
                "name": "Tuned Gradient Boosting",
                "accuracy": 93.25,
                "precision": 87.69,
                "recall": 83.47,
                "f1": 85.53,
                "roc_auc": 97.79
            },
            {
                "name": "Random Forest",
                "accuracy": 91.50,
                "precision": 87.93,
                "recall": 74.69,
                "f1": 80.77,
                "roc_auc": 96.57
            }
        ]

        # =========================================================
        # THRESHOLD ANALYSIS
        # =========================================================

        threshold_data = {
            "thresholds": [
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70
            ],

            "accuracy": [
                92.05,
                92.52,
                92.95,
                93.28,
                93.32,
                93.35,
                92.98,
                93.00,
                92.65
            ],

            "precision": [
                78.89,
                81.08,
                83.50,
                86.04,
                87.57,
                88.94,
                89.85,
                91.63,
                93.21
            ],

            "recall": [
                91.11,
                89.64,
                87.87,
                85.77,
                84.00,
                82.43,
                79.60,
                77.82,
                74.69
            ],

            "f1": [
                84.56,
                85.15,
                85.63,
                85.91,
                85.74,
                85.56,
                84.41,
                84.16,
                82.93
            ]
        }

        # =========================================================
        # TOP LOGISTIC REGRESSION COEFFICIENTS
        # =========================================================

        coefficients = [
            {
                "feature": "Total Debt To Income Ratio",
                "value": -6.388053
            },
            {
                "feature": "Annual Income",
                "value": 2.131762
            },
            {
                "feature": "Net Worth",
                "value": 1.628429
            },
            {
                "feature": "Loan Duration",
                "value": -1.558705
            },
            {
                "feature": "Education Level - Doctorate",
                "value": 1.477715
            },
            {
                "feature": "Loan Amount",
                "value": -1.457274
            },
            {
                "feature": "Education Level - High School",
                "value": -1.398221
            },
            {
                "feature": "Employment Status - Unemployed",
                "value": -1.000442
            },
            {
                "feature": "Length Of Credit History",
                "value": 0.935459
            },
            {
                "feature": "Bankruptcy History",
                "value": -0.822369
            },
            {
                "feature": "Education Level - Master",
                "value": 0.739162
            },
            {
                "feature": "Previous Loan Defaults",
                "value": -0.708392
            },
            {
                "feature": "Education Level - Associate",
                "value": -0.677022
            },
            {
                "feature": "Employment Status - Self-Employed",
                "value": 0.530786
            },
            {
                "feature": "Employment Status - Employed",
                "value": 0.499613
            },
            {
                "feature": "Marital Status - Widowed",
                "value": -0.413316
            },
            {
                "feature": "Home Ownership - Mortgage",
                "value": 0.364657
            },
            {
                "feature": "Home Ownership - Other",
                "value": -0.332332
            },
            {
                "feature": "Home Ownership - Rent",
                "value": -0.312818
            },
            {
                "feature": "Home Ownership - Own",
                "value": 0.310450
            }
        ]

        # =========================================================
        # TOP NUMERICAL CORRELATIONS
        # FROM NOTEBOOK
        # =========================================================

        correlation_features = [
            "MonthlyIncome",
            "AnnualIncome",
            "NetWorth",
            "TotalAssets",
            "MonthlyLoanPayment",
            "LoanAmount",
            "BaseInterestRate",
            "InterestRate",
            "TotalDebtToIncomeRatio",
            "RiskScore"
        ]

        correlation_values = [
            0.60,
            0.60,
            0.19,
            0.18,
            -0.19,
            -0.24,
            -0.25,
            -0.30,
            -0.41,
            -0.77
        ]

        # =========================================================
        # POTENTIAL LEAKAGE CORRELATIONS
        # =========================================================

        leakage_labels = [
            "RiskScore",
            "InterestRate",
            "BaseInterestRate",
            "MonthlyLoanPayment",
            "TotalDebtToIncomeRatio"
        ]

        leakage_matrix = [
            [1.00, 0.27, 0.26, 0.12, 0.34],
            [0.27, 1.00, 0.83, 0.13, 0.13],
            [0.26, 0.83, 1.00, 0.12, 0.13],
            [0.12, 0.13, 0.12, 1.00, 0.57],
            [0.34, 0.13, 0.13, 0.57, 1.00]
        ]

        # =========================================================
        # RISK SCORE BOX PLOT SUMMARY
        # APPROXIMATION BASED ON NOTEBOOK VISUAL
        # =========================================================

        risk_score_data = {
            "not_approved": {
                "min": 40,
                "q1": 50,
                "median": 54,
                "q3": 57,
                "max": 67
            },

            "approved": {
                "min": 30,
                "q1": 37.5,
                "median": 40,
                "q3": 42,
                "max": 49.5
            }
        }

        # =========================================================
        # RENDER
        # =========================================================

        return render_template(
            "model.html",

            metrics=metrics,

            total_records=total_records,
            total_features=total_features,
            approved_count=approved_count,
            not_approved_count=not_approved_count,

            confusion_matrix=confusion_matrix,

            model_comparison=model_comparison,

            threshold_data=threshold_data,

            coefficients=coefficients,

            correlation_features=correlation_features,
            correlation_values=correlation_values,

            leakage_labels=leakage_labels,
            leakage_matrix=leakage_matrix,

            risk_score_data=risk_score_data,

            preview_data=preview_data
        )

    except Exception as e:

        print("Model Insights Error:", e)

        return render_template(
            "model.html",

            metrics={
                "accuracy": 93.33,
                "precision": 87.57,
                "recall": 84.00,
                "f1": 85.74,
                "roc_auc": 97.86
            },

            total_records=20000,
            total_features=36,
            approved_count=4780,
            not_approved_count=15220,

            confusion_matrix={
                "tn": 2911,
                "fp": 133,
                "fn": 136,
                "tp": 820
            },

            model_comparison=[],

            threshold_data={
                "thresholds": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": []
            },

            coefficients=[],

            correlation_features=[],
            correlation_values=[],

            leakage_labels=[],
            leakage_matrix=[],

            risk_score_data={
                "not_approved": {
                    "min": 40,
                    "q1": 50,
                    "median": 54,
                    "q3": 57,
                    "max": 67
                },
                "approved": {
                    "min": 30,
                    "q1": 37.5,
                    "median": 40,
                    "q3": 42,
                    "max": 49.5
                }
            },

            preview_data=[]
        )
# =========================================================
# ABOUT PAGE
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )