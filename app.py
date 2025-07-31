from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

# Load your saved model and columns
with open('models/logistic_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Raw input from form
    raw = request.form

    # Basic inputs
    dependents = int(raw.get('Dependents'))
    education = 1 if raw.get('Education') == 'Graduate' else 0
    applicant_income = float(raw.get('ApplicantIncome'))
    coapplicant_income = float(raw.get('CoapplicantIncome'))
    loan_amount = float(raw.get('LoanAmount'))
    loan_term = float(raw.get('Loan_Amount_Term'))

    # Log transforms
    applicant_income_log = np.log1p(applicant_income)
    coapplicant_income_log = np.log1p(coapplicant_income)
    loan_amount_log = np.log1p(loan_amount)

    # One-hot like inputs
    gender_female = 1 if raw.get('Gender') == 'Female' else 0
    gender_male = 1 - gender_female

    married_yes = 1 if raw.get('Married') == 'Yes' else 0
    married_no = 1 - married_yes

    self_employed_yes = 1 if raw.get('Self_Employed') == 'Yes' else 0
    self_employed_no = 1 - self_employed_yes

    area = raw.get('Property_Area')
    area_rural = 1 if area == 'Rural' else 0
    area_semiurban = 1 if area == 'Semiurban' else 0
    area_urban = 1 if area == 'Urban' else 0

    credit = raw.get('Credit_History')
    credit_1 = 1 if credit == 'Known' else 0
    credit_0 = 1 if credit == 'No Credit History' else 0
    credit_unknown = 1 if credit == 'Unknown' else 0

    # Build dictionary of features
    input_dict = {
        'Dependents': dependents,
        'Education': education,
        'ApplicantIncome_log': applicant_income_log,
        'LoanAmount_log': loan_amount_log,
        'CoapplicantIncome_log': coapplicant_income_log,
        'Loan_Amount_Term': loan_term,
        'Gender_Female': gender_female,
        'Gender_Male': gender_male,
        'Married_No': married_no,
        'Married_Yes': married_yes,
        'Self_Employed_No': self_employed_no,
        'Self_Employed_Yes': self_employed_yes,
        'Property_Area_Rural': area_rural,
        'Property_Area_Semiurban': area_semiurban,
        'Property_Area_Urban': area_urban,
        'Credit_History_0.0': credit_0,
        'Credit_History_1.0': credit_1,
        'Credit_History_Unknown': credit_unknown
    }

    # Ensure all expected columns exist
    full_input = {col: input_dict.get(col, 0) for col in model_columns}
    input_df = pd.DataFrame([full_input], columns=model_columns)

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "✅ Eligible for Loan (Y)" if prediction == 1 else "❌ Not Eligible for Loan (N)"
    confidence = f"{probability * 100:.2f}% confidence"     

    return render_template('result.html', prediction=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
