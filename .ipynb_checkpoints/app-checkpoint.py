from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime
import json

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


# Load trained Logistic Regression pipeline
model = joblib.load(MODEL_PATH)


# =========================================================
# HIDDEN FEATURE DEFAULTS
# =========================================================
# These values were calculated from X_train using median
# values for numerical features.
#
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

        # -------------------------------------------------
        # Application Date
        # -------------------------------------------------

        application_date = request.form[
            "ApplicationDate"
        ]

        date_obj = datetime.strptime(
            application_date,
            "%Y-%m-%d"
        )

        application_year = date_obj.year

        application_month = date_obj.month

        application_day_of_week = date_obj.weekday()


        # -------------------------------------------------
        # User Input
        # -------------------------------------------------

        input_data = {

            # Personal Information
            "Age":
                int(request.form["Age"]),

            "AnnualIncome":
                int(request.form["AnnualIncome"]),

            "CreditScore":
                int(request.form["CreditScore"]),

            "EmploymentStatus":
                request.form["EmploymentStatus"],

            "EducationLevel":
                request.form["EducationLevel"],

            "Experience":
                int(request.form["Experience"]),

            "LoanAmount":
                int(request.form["LoanAmount"]),

            "LoanDuration":
                int(request.form["LoanDuration"]),

            "MaritalStatus":
                request.form["MaritalStatus"],

            "NumberOfDependents":
                int(request.form["NumberOfDependents"]),

            "HomeOwnershipStatus":
                request.form["HomeOwnershipStatus"],


            # Financial Information
            "MonthlyDebtPayments":
                int(request.form["MonthlyDebtPayments"]),

            "SavingsAccountBalance":
                int(request.form[
                    "SavingsAccountBalance"
                ]),

            "NetWorth":
                int(request.form["NetWorth"]),


            # Credit Information
            "BankruptcyHistory":
                int(request.form[
                    "BankruptcyHistory"
                ]),

            "PreviousLoanDefaults":
                int(request.form[
                    "PreviousLoanDefaults"
                ]),

            "LengthOfCreditHistory":
                int(request.form[
                    "LengthOfCreditHistory"
                ]),


            # Loan Information
            "LoanPurpose":
                request.form["LoanPurpose"],


            # -------------------------------------------------
            # Automatically generated date features
            # -------------------------------------------------

            "ApplicationYear":
                application_year,

            "ApplicationMonth":
                application_month,

            "ApplicationDayOfWeek":
                application_day_of_week,


            # -------------------------------------------------
            # Hidden model features
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

        probability = model.predict_proba(
            input_df
        )[0][1]


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
        # This is an application-level indicator based
        # on predicted approval probability.
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

            submitted=True

        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

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

    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------
    # Basic KPIs
    # ---------------------------------------------

    total_applications = len(df)

    approved = int(df["LoanApproved"].sum())

    rejected = total_applications - approved

    approval_rate = round(
        approved / total_applications * 100,
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


    # ---------------------------------------------
    # Approval Distribution
    # ---------------------------------------------

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


    # ---------------------------------------------
    # Approval by Employment
    # ---------------------------------------------

    employment = (
        df.groupby("EmploymentStatus")["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )

    approval_by_employment = {
        "labels": employment.index.tolist(),
        "values": employment.values.tolist()
    }


    # ---------------------------------------------
    # Approval by Education
    # ---------------------------------------------

    education = (
        df.groupby("EducationLevel")["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )

    approval_by_education = {
        "labels": education.index.tolist(),
        "values": education.values.tolist()
    }


    # ---------------------------------------------
    # Approval by Home Ownership
    # ---------------------------------------------

    home = (
        df.groupby("HomeOwnershipStatus")["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )

    approval_by_home = {
        "labels": home.index.tolist(),
        "values": home.values.tolist()
    }


    # ---------------------------------------------
    # Approval by Loan Purpose
    # ---------------------------------------------

    purpose = (
        df.groupby("LoanPurpose")["LoanApproved"]
        .mean()
        .mul(100)
        .round(2)
    )

    approval_by_purpose = {
        "labels": purpose.index.tolist(),
        "values": purpose.values.tolist()
    }


    # ---------------------------------------------
    # Credit Score Distribution
    # ---------------------------------------------

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


    # ---------------------------------------------
    # Income
    # ---------------------------------------------

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


    # ---------------------------------------------
    # Loan Amount
    # ---------------------------------------------

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


    # ---------------------------------------------
    # Debt-to-Income Ratio
    # ---------------------------------------------

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


    # ---------------------------------------------
    # Previous Defaults
    # ---------------------------------------------

    defaults = (
        df.groupby("PreviousLoanDefaults")["LoanApproved"]
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
            defaults.get(0, 0),
            defaults.get(1, 0)
        ]
    }


    # ---------------------------------------------
    # Bankruptcy
    # ---------------------------------------------

    bankruptcy = (
        df.groupby("BankruptcyHistory")["LoanApproved"]
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
            bankruptcy.get(0, 0),
            bankruptcy.get(1, 0)
        ]
    }


    return render_template(

        "analytics.html",

        total_applications=total_applications,

        approved=approved,

        rejected=rejected,

        approval_rate=approval_rate,

        avg_loan_amount=avg_loan_amount,

        avg_credit_score=avg_credit_score,

        avg_income=avg_income,

        approval_distribution=json.dumps(
            approval_distribution
        ),

        approval_by_employment=json.dumps(
            approval_by_employment
        ),

        approval_by_education=json.dumps(
            approval_by_education
        ),

        approval_by_home=json.dumps(
            approval_by_home
        ),

        approval_by_purpose=json.dumps(
            approval_by_purpose
        ),

        credit_score=json.dumps(
            credit_score
        ),

        income=json.dumps(
            income
        ),

        loan_amount=json.dumps(
            loan_amount
        ),

        dti=json.dumps(
            dti
        ),

        approval_by_defaults=json.dumps(
            approval_by_defaults
        ),

        approval_by_bankruptcy=json.dumps(
            approval_by_bankruptcy
        )
    )
# =========================================================
# MODEL INSIGHTS PAGE
# =========================================================

@app.route("/model")
def model_insights():

    return render_template(
        "model.html"
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