# model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load application and credit record datasets"""
    print("Loading datasets...")
    application = pd.read_csv('data/application_record.csv')
    credit = pd.read_csv('data/credit_record.csv')
    
    print(f"Application records: {len(application)}")
    print(f"Credit records: {len(credit)}")
    
    return application, credit


def create_target_variable(credit):
    """
    Create target variable from credit record
    STATUS meanings:
        0: 1-29 days past due
        1: 30-59 days past due
        2: 60-89 days past due
        3: 90-119 days past due
        4: 120-149 days past due
        5: Overdue or bad debts
        C: Paid off that month
        X: No loan for the month
    
    Target: 1 = Good (Approved), 0 = Bad (Rejected)
    """
    print("Creating target variable...")
    
    # Convert STATUS to numeric risk score
    status_map = {
        'C': 0,  # Paid off - good
        'X': 0,  # No loan - neutral (good)
        '0': 1,  # 1-29 days late
        '1': 2,  # 30-59 days late
        '2': 3,  # 60-89 days late
        '3': 4,  # 90-119 days late
        '4': 5,  # 120-149 days late
        '5': 6   # Bad debt
    }
    
    credit['STATUS'] = credit['STATUS'].astype(str)
    credit['status_score'] = credit['STATUS'].map(status_map)
    
    # Aggregate by ID - calculate max risk score and count of records
    credit_agg = credit.groupby('ID').agg({
        'status_score': ['max', 'mean'],
        'MONTHS_BALANCE': 'count'
    }).reset_index()
    
    credit_agg.columns = ['ID', 'max_status', 'avg_status', 'num_records']
    
    # Define target: Good if max_status <= 1 (never more than 30 days late)
    credit_agg['target'] = (credit_agg['max_status'] <= 1).astype(int)
    
    print(f"Target distribution:\n{credit_agg['target'].value_counts(normalize=True)}")
    
    return credit_agg


def merge_datasets(application, credit_agg):
    """Merge application and credit records"""
    print("Merging datasets...")
    
    data = application.merge(credit_agg, on='ID', how='inner')
    print(f"Merged dataset size: {len(data)}")
    
    return data


def drop_unwanted_features(data):
    """Drop features that are not useful for prediction"""
    print("Dropping unwanted features...")
    
    # Columns to drop
    drop_cols = [
        'ID',           # Identifier, not predictive
        'FLAG_MOBIL',   # Almost everyone has mobile (constant)
        'max_status',   # Used to create target
        'avg_status',   # Used to create target
        'num_records'   # Leakage - info from credit record
    ]
    
    # Drop only columns that exist
    drop_cols = [col for col in drop_cols if col in data.columns]
    data = data.drop(columns=drop_cols)
    
    print(f"Remaining features: {list(data.columns)}")
    
    return data


def handle_missing_values(data):
    """Handle missing values in the dataset"""
    print("Handling missing values...")
    print(f"Missing values before:\n{data.isnull().sum()}")
    
    # OCCUPATION_TYPE has missing values - fill with 'Unknown'
    if 'OCCUPATION_TYPE' in data.columns:
        data['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE'].fillna('Unknown')
    
    # For numeric columns, fill with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
    
    # For categorical columns, fill with mode
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    print(f"Missing values after:\n{data.isnull().sum().sum()}")
    
    return data


def feature_engineering(data):
    """Create new features from existing ones"""
    print("Feature engineering...")
    
    # Convert DAYS_BIRTH to age in years (DAYS_BIRTH is negative)
    if 'DAYS_BIRTH' in data.columns:
        data['AGE_YEARS'] = (-data['DAYS_BIRTH']) / 365
        data = data.drop(columns=['DAYS_BIRTH'])
    
    # Convert DAYS_EMPLOYED to years employed
    # Positive values (365243) indicate unemployed/pensioner
    if 'DAYS_EMPLOYED' in data.columns:
        data['EMPLOYED_YEARS'] = data['DAYS_EMPLOYED'].apply(
            lambda x: 0 if x > 0 else -x / 365
        )
        data['IS_EMPLOYED'] = (data['DAYS_EMPLOYED'] < 0).astype(int)
        data = data.drop(columns=['DAYS_EMPLOYED'])
    
    # Income per family member
    if 'AMT_INCOME_TOTAL' in data.columns and 'CNT_FAM_MEMBERS' in data.columns:
        data['INCOME_PER_MEMBER'] = data['AMT_INCOME_TOTAL'] / (data['CNT_FAM_MEMBERS'] + 1)
    
    # Income to children ratio
    if 'AMT_INCOME_TOTAL' in data.columns and 'CNT_CHILDREN' in data.columns:
        data['INCOME_PER_CHILD'] = data['AMT_INCOME_TOTAL'] / (data['CNT_CHILDREN'] + 1)
    
    # Has contact flags combined
    contact_flags = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
    existing_flags = [f for f in contact_flags if f in data.columns]
    if existing_flags:
        data['CONTACT_SCORE'] = data[existing_flags].sum(axis=1)
    
    print(f"Features after engineering: {list(data.columns)}")
    
    return data


def handle_categorical_variables(data):
    """Encode categorical variables"""
    print("Encoding categorical variables...")
    
    # Identify categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {cat_cols}")
    
    # Use one-hot encoding for categorical variables
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    print(f"Shape after encoding: {data.shape}")
    
    return data


def split_data(data, target_col='target', test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print("Splitting data...")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)}, Testing set: {len(X_test)}")
    print(f"Target distribution in train:\n{y_train.value_counts(normalize=True)}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale numerical features"""
    print("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING MODELS")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training {name}...")
        print('='*40)
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate threshold-based metrics
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_threshold = 0.5
        best_f1 = f1
        
        for thresh in thresholds:
            y_pred_thresh = (y_prob >= thresh).astype(int)
            f1_thresh = f1_score(y_test, y_pred_thresh)
            if f1_thresh > best_f1:
                best_f1 = f1_thresh
                best_threshold = thresh
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Best Threshold': best_threshold,
            'Best F1': best_f1
        })
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Best Threshold: {best_threshold}")
        print(f"Best F1 at threshold: {best_f1:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return results, trained_models


def select_best_model(results, trained_models):
    """Select the best model based on F1 score"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Select best model based on F1 Score
    best_idx = results_df['F1 Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = trained_models[best_model_name]
    best_threshold = results_df.loc[best_idx, 'Best Threshold']
    
    print(f"\n{'='*40}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"F1 Score: {results_df.loc[best_idx, 'F1 Score']:.4f}")
    print(f"Optimal Threshold: {best_threshold}")
    print('='*40)
    
    return best_model, best_model_name, best_threshold


def save_artifacts(model, scaler, feature_columns, threshold):
    """Save model and preprocessing artifacts"""
    print("\nSaving artifacts...")
    
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    with open('threshold.pkl', 'wb') as f:
        pickle.dump(threshold, f)
    
    print("Artifacts saved successfully!")


def main():
    """Main training pipeline"""
    print("="*60)
    print("CREDIT CARD APPROVAL PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    application, credit = load_data()
    
    # Create target variable
    credit_agg = create_target_variable(credit)
    
    # Merge datasets
    data = merge_datasets(application, credit_agg)
    
    # Drop unwanted features
    data = drop_unwanted_features(data)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Feature engineering
    data = feature_engineering(data)
    
    # Handle categorical variables
    data = handle_categorical_variables(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Save feature columns for later use
    feature_columns = X_train.columns.tolist()
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train and evaluate models
    results, trained_models = train_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Select best model
    best_model, best_model_name, best_threshold = select_best_model(
        results, trained_models
    )
    
    # Save artifacts
    save_artifacts(best_model, scaler, feature_columns, best_threshold)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return best_model, scaler, feature_columns


if __name__ == "__main__":
    main()
