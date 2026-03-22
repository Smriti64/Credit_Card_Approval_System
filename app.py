# app.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load saved artifacts
def load_artifacts():
    """Load trained model and preprocessing artifacts"""
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    with open('threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    
    return model, scaler, feature_columns, threshold

# Load artifacts at startup
model, scaler, feature_columns, threshold = load_artifacts()


def preprocess_input(form_data):
    """Preprocess user input to match training data format"""
    
    # Create a dictionary with raw input
    data = {
        'CODE_GENDER': form_data.get('gender'),
        'FLAG_OWN_CAR': form_data.get('own_car'),
        'FLAG_OWN_REALTY': form_data.get('own_realty'),
        'CNT_CHILDREN': int(form_data.get('children', 0)),
        'AMT_INCOME_TOTAL': float(form_data.get('income', 0)),
        'NAME_INCOME_TYPE': form_data.get('income_type'),
        'NAME_EDUCATION_TYPE': form_data.get('education'),
        'NAME_FAMILY_STATUS': form_data.get('family_status'),
        'NAME_HOUSING_TYPE': form_data.get('housing_type'),
        'DAYS_BIRTH': -int(form_data.get('age', 30)) * 365,
        'DAYS_EMPLOYED': -int(form_data.get('employed_years', 0)) * 365 if int(form_data.get('employed_years', 0)) > 0 else 365243,
        'FLAG_WORK_PHONE': int(form_data.get('work_phone', 0)),
        'FLAG_PHONE': int(form_data.get('phone', 0)),
        'FLAG_EMAIL': int(form_data.get('email', 0)),
        'OCCUPATION_TYPE': form_data.get('occupation', 'Unknown'),
        'CNT_FAM_MEMBERS': int(form_data.get('family_members', 1))
    }
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Apply same feature engineering as training
    # Age in years
    df['AGE_YEARS'] = (-df['DAYS_BIRTH']) / 365
    df = df.drop(columns=['DAYS_BIRTH'])
    
    # Employed years
    df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'].apply(
        lambda x: 0 if x > 0 else -x / 365
    )
    df['IS_EMPLOYED'] = (df['DAYS_EMPLOYED'] < 0).astype(int)
    df = df.drop(columns=['DAYS_EMPLOYED'])
    
    # Income per family member
    df['INCOME_PER_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    
    # Income per child
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
    
    # Contact score
    df['CONTACT_SCORE'] = df['FLAG_WORK_PHONE'] + df['FLAG_PHONE'] + df['FLAG_EMAIL']
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Ensure all required columns exist (fill missing with 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select and order columns to match training
    df = df[feature_columns]
    
    return df


@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/apply')
def application():
    """Application form page"""
    return render_template('application.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Process application and show result"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Preprocess input
        processed_data = preprocess_input(form_data)
        
        # Scale features
        scaled_data = scaler.transform(processed_data)
        
        # Get prediction probability
        probability = model.predict_proba(scaled_data)[0][1]
        
        # Apply threshold
        approved = probability >= threshold
        
        # Determine status and confidence
        if approved:
            status = "APPROVED"
            confidence = probability * 100
        else:
            status = "REJECTED"
            confidence = (1 - probability) * 100
        
        # Prepare result data
        result_data = {
            'status': status,
            'probability': round(probability * 100, 2),
            'confidence': round(confidence, 2),
            'income': float(form_data.get('income', 0)),
            'age': int(form_data.get('age', 0)),
            'employed_years': int(form_data.get('employed_years', 0)),
            'education': form_data.get('education', ''),
            'occupation': form_data.get('occupation', '')
        }
        
        return render_template('result.html', result=result_data)
    
    except Exception as e:
        return render_template('result.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
