# app.py — AprovAi Credit Card Approval System (Improved)
#
# Improvements over v1:
#   • Credit-limit regression prediction
#   • Structured JSON API endpoint (/api/predict)
#   • Input validation with clear error messages
#   • Request logging (inputs + results)
#   • XAI — top-5 feature importance displayed on result page
#   • Clean error handling (user-friendly messages)
#   • Production-style structure

import os
import logging
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
import pickle

from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts():
    """Load all saved ML artifacts. Raises clear error if files are missing."""
    required = ['best_model.pkl', 'scaler.pkl', 'feature_columns.pkl', 'threshold.pkl']
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artifact: {path}. Please run model_training.py first."
            )

    with open('best_model.pkl',      'rb') as f: clf            = pickle.load(f)
    with open('scaler.pkl',          'rb') as f: scaler         = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f: feature_columns= pickle.load(f)
    with open('threshold.pkl',       'rb') as f: threshold      = pickle.load(f)

    # Credit-limit regressor is optional (falls back gracefully)
    reg = None
    if os.path.exists('credit_limit_model.pkl'):
        with open('credit_limit_model.pkl', 'rb') as f:
            reg = pickle.load(f)

    log.info("All ML artifacts loaded successfully.")
    return clf, reg, scaler, feature_columns, threshold


clf, credit_reg, scaler, feature_columns, threshold = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

VALID_GENDERS      = {'M', 'F'}
VALID_YN           = {'Y', 'N'}
VALID_INCOME_TYPES = {
    'Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'
}
VALID_EDU = {
    'Lower secondary', 'Secondary / secondary special',
    'Incomplete higher', 'Higher education', 'Academic degree'
}
VALID_FAMILY_STATUS = {
    'Single / not married', 'Married', 'Civil marriage', 'Separated', 'Widow'
}
VALID_HOUSING = {
    'House / apartment', 'With parents', 'Municipal apartment',
    'Rented apartment', 'Office apartment', 'Co-op apartment'
}


def validate_form(form) -> list:
    """Return a list of validation error messages (empty = valid)."""
    errors = []

    def require(field, label):
        val = form.get(field, '').strip()
        if not val:
            errors.append(f"{label} is required.")
        return val

    gender = require('gender', 'Gender')
    if gender and gender not in VALID_GENDERS:
        errors.append("Invalid gender value.")

    own_car = require('own_car', 'Own Car')
    if own_car and own_car not in VALID_YN:
        errors.append("Invalid car ownership value.")

    own_realty = require('own_realty', 'Own Realty')
    if own_realty and own_realty not in VALID_YN:
        errors.append("Invalid realty ownership value.")

    income_type = require('income_type', 'Income Type')
    if income_type and income_type not in VALID_INCOME_TYPES:
        errors.append("Invalid income type.")

    education = require('education', 'Education')
    if education and education not in VALID_EDU:
        errors.append("Invalid education level.")

    family_status = require('family_status', 'Marital Status')
    if family_status and family_status not in VALID_FAMILY_STATUS:
        errors.append("Invalid marital status.")

    housing = require('housing_type', 'Housing Type')
    if housing and housing not in VALID_HOUSING:
        errors.append("Invalid housing type.")

    # Numeric validations
    try:
        age = int(form.get('age', 0))
        if not (18 <= age <= 100):
            errors.append("Age must be between 18 and 100.")
    except ValueError:
        errors.append("Age must be a valid number.")

    try:
        income = float(form.get('income', 0))
        if income <= 0:
            errors.append("Income must be greater than 0.")
    except ValueError:
        errors.append("Income must be a valid number.")

    try:
        children = int(form.get('children', 0))
        if children < 0:
            errors.append("Children count cannot be negative.")
    except ValueError:
        errors.append("Children count must be a valid number.")

    try:
        employed = int(form.get('employed_years', 0))
        if employed < 0:
            errors.append("Employed years cannot be negative.")
    except ValueError:
        errors.append("Employed years must be a valid number.")

    try:
        fam = int(form.get('family_members', 1))
        if fam < 1:
            errors.append("Family members must be at least 1.")
    except ValueError:
        errors.append("Family members must be a valid number.")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_input(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form submission into the same feature space used during training.
    Mirrors all steps in model_training.py's feature_engineering function.
    """
    age           = int(form_data.get('age', 30))
    employed_years= int(form_data.get('employed_years', 0))
    income        = float(form_data.get('income', 0))
    children      = int(form_data.get('children', 0))
    fam_members   = int(form_data.get('family_members', 1))

    raw = {
        'CODE_GENDER':        form_data.get('gender'),
        'FLAG_OWN_CAR':       form_data.get('own_car'),
        'FLAG_OWN_REALTY':    form_data.get('own_realty'),
        'CNT_CHILDREN':       children,
        'AMT_INCOME_TOTAL':   income,
        'NAME_INCOME_TYPE':   form_data.get('income_type'),
        'NAME_EDUCATION_TYPE':form_data.get('education'),
        'NAME_FAMILY_STATUS': form_data.get('family_status'),
        'NAME_HOUSING_TYPE':  form_data.get('housing_type'),
        # Convert to DAYS format used during training
        'DAYS_BIRTH':         -age * 365,
        'DAYS_EMPLOYED':      -employed_years * 365 if employed_years > 0 else 365243,
        'FLAG_WORK_PHONE':    int(form_data.get('work_phone', 0)),
        'FLAG_PHONE':         int(form_data.get('phone', 0)),
        'FLAG_EMAIL':         int(form_data.get('email', 0)),
        'OCCUPATION_TYPE':    form_data.get('occupation', 'Unknown'),
        'CNT_FAM_MEMBERS':    fam_members,
    }

    df = pd.DataFrame([raw])

    # ── Replicate feature_engineering steps ───────────────────────────────
    df['AGE_YEARS']   = age
    df.drop(columns=['DAYS_BIRTH'], inplace=True)

    df['EMPLOYED_YEARS'] = employed_years
    df['IS_EMPLOYED']    = int(employed_years > 0)
    df.drop(columns=['DAYS_EMPLOYED'], inplace=True)

    df['INCOME_PER_MEMBER'] = income / (fam_members + 1)
    df['INCOME_PER_CHILD']  = income / (children + 1)
    df['CONTACT_SCORE']     = (
        int(form_data.get('work_phone', 0)) +
        int(form_data.get('phone', 0)) +
        int(form_data.get('email', 0))
    )

    # One-hot encode (must match training encoding)
    df = pd.get_dummies(df, drop_first=True)

    # Align to training feature set (fill missing dummies with 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]   # exact column order matters for scaler


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance helper (XAI)
# ─────────────────────────────────────────────────────────────────────────────

def get_top_features(top_n: int = 5) -> list[dict]:
    """Return top-N feature importances from the classifier (if available)."""
    if not hasattr(clf, 'feature_importances_'):
        return []
    imp = pd.Series(clf.feature_importances_, index=feature_columns)
    top = imp.nlargest(top_n)
    return [
        {'name': k.replace('_', ' ').title(), 'importance': round(float(v) * 100, 1)}
        for k, v in top.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/apply')
def application():
    return render_template('application.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Web form submission → result page."""
    try:
        form_data = request.form.to_dict()

        # Validate
        errors = validate_form(request.form)
        if errors:
            return render_template('result.html', error=' | '.join(errors))

        result = _run_prediction(form_data)

        # Log the request (no PII stored permanently; log is for debugging)
        log.info(
            f"PREDICTION | income={form_data.get('income')} | "
            f"age={form_data.get('age')} | "
            f"status={result['status']} | "
            f"confidence={result['confidence']}%"
        )

        return render_template('result.html', result=result)

    except Exception as e:
        log.error(f"Prediction error: {traceback.format_exc()}")
        return render_template('result.html',
                               error="An unexpected error occurred. Please try again.")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint.

    Request body (JSON):
    {
        "gender": "M", "own_car": "Y", "own_realty": "N",
        "children": 1, "income": 75000, "income_type": "Working",
        "education": "Higher education", "family_status": "Married",
        "housing_type": "House / apartment", "age": 35,
        "employed_years": 5, "work_phone": 1, "phone": 1, "email": 0,
        "occupation": "Managers", "family_members": 3
    }

    Response:
    {
        "approval_status": "Approved",
        "confidence_score": 87.4,
        "predicted_credit_limit": 18500,
        "approval_probability": 0.874,
        "top_features": [{"name": "...", "importance": ...}]
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'Invalid or empty JSON body.'}), 400

        result = _run_prediction(data)

        return jsonify({
            'approval_status':       result['status'].title(),
            'confidence_score':      result['confidence'],
            'predicted_credit_limit': result.get('credit_limit', None),
            'approval_probability':  round(result['probability'] / 100, 4),
            'top_features':          result.get('top_features', []),
        })

    except Exception as e:
        log.error(f"API error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error.'}), 500




def _run_prediction(form_data: dict) -> dict:
    """Preprocess → scale → classify → (optionally) regress → return result dict."""

    processed   = preprocess_input(form_data)
    scaled      = scaler.transform(processed)

    probability = float(clf.predict_proba(scaled)[0][1])
    approved    = probability >= threshold
    status      = 'APPROVED' if approved else 'REJECTED'
    confidence  = probability * 100 if approved else (1 - probability) * 100

    # Credit limit prediction
    credit_limit = None
    if credit_reg is not None:
        raw_limit    = float(credit_reg.predict(scaled)[0])
        credit_limit = int(round(raw_limit / 100) * 100)   # round to $100
        if not approved:
            credit_limit = None   # only show limit if approved

    result = {
        'status':       status,
        'probability':  round(probability * 100, 2),
        'confidence':   round(confidence, 2),
        'income':       float(form_data.get('income', 0)),
        'age':          int(form_data.get('age', 0)),
        'employed_years': int(form_data.get('employed_years', 0)),
        'education':    form_data.get('education', ''),
        'occupation':   form_data.get('occupation', ''),
        'credit_limit': credit_limit,
        'top_features': get_top_features(5),
    }
    return result



@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': clf is not None,
        'regressor_loaded': credit_reg is not None,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
