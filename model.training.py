# model_training.py — AprovAi Credit Card Approval System

import os
import logging
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(app_path='data/application_record.csv',
              credit_path='data/credit_record.csv'):
    """Load application and credit record CSV files."""
    log.info("Loading datasets …")
    application = pd.read_csv(app_path)
    credit      = pd.read_csv(credit_path)
    log.info(f"Application rows: {len(application):,}  |  Credit rows: {len(credit):,}")
    return application, credit


# ─────────────────────────────────────────────────────────────────────────────
# TARGET ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def create_target_variable(credit: pd.DataFrame):
    """
    Build binary approval target and synthetic credit-limit proxy.

    STATUS codes
    ─────────────
    C  = paid off that month   → risk 0
    X  = no loan               → risk 0
    0  = 1-29 days overdue     → risk 1
    1  = 30-59 days overdue    → risk 2
    2  = 60-89 days overdue    → risk 3
    3  = 90-119 days overdue   → risk 4
    4  = 120-149 days overdue  → risk 5
    5  = bad debt              → risk 6

    Approval target  : 1 (Good) if max_risk ≤ 1, else 0 (Bad)
    Credit limit     : synthetic value derived from income (added at merge step)
    """
    log.info("Creating target variable …")
    status_map = {'C': 0, 'X': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6}

    credit = credit.copy()
    credit['STATUS'] = credit['STATUS'].astype(str)
    credit['status_score'] = credit['STATUS'].map(status_map)

    agg = credit.groupby('ID').agg(
        max_status=('status_score', 'max'),
        avg_status=('status_score', 'mean'),
        num_records=('MONTHS_BALANCE', 'count')
    ).reset_index()

    # Classification target: good payer = 1, bad payer = 0
    agg['target'] = (agg['max_status'] <= 1).astype(int)

    dist = agg['target'].value_counts(normalize=True)
    log.info(f"Target distribution → Good: {dist.get(1, 0):.1%}  Bad: {dist.get(0, 0):.1%}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_clean(application: pd.DataFrame, credit_agg: pd.DataFrame):
    """Merge, drop leakage columns, handle nulls."""
    log.info("Merging datasets …")
    data = application.merge(credit_agg, on='ID', how='inner')
    log.info(f"Merged size: {len(data):,}")

    # Drop identifier / leakage / near-constant columns
    drop_cols = ['ID', 'FLAG_MOBIL', 'max_status', 'avg_status', 'num_records']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # Fill missing OCCUPATION_TYPE
    if 'OCCUPATION_TYPE' in data.columns:
        data['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE'].fillna('Unknown')

    # Fill numeric nulls with median, categorical with mode
    for col in data.select_dtypes(include=np.number).columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)
    for col in data.select_dtypes(include='object').columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].mode()[0], inplace=True)

    log.info(f"Null values remaining: {data.isnull().sum().sum()}")
    return data


def feature_engineering(data: pd.DataFrame):
    """Derive new features; also create the synthetic credit-limit target."""
    log.info("Feature engineering …")

    # Age from DAYS_BIRTH (negative integer)
    if 'DAYS_BIRTH' in data.columns:
        data['AGE_YEARS'] = (-data['DAYS_BIRTH']) / 365
        data.drop(columns=['DAYS_BIRTH'], inplace=True)

    # Employment years; 365243 is the sentinel for unemployed
    if 'DAYS_EMPLOYED' in data.columns:
        data['EMPLOYED_YEARS'] = data['DAYS_EMPLOYED'].apply(
            lambda x: 0 if x > 0 else -x / 365
        )
        data['IS_EMPLOYED'] = (data['DAYS_EMPLOYED'] < 0).astype(int)
        data.drop(columns=['DAYS_EMPLOYED'], inplace=True)

    # Financial ratios
    if {'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS'}.issubset(data.columns):
        data['INCOME_PER_MEMBER'] = data['AMT_INCOME_TOTAL'] / (data['CNT_FAM_MEMBERS'] + 1)
    if {'AMT_INCOME_TOTAL', 'CNT_CHILDREN'}.issubset(data.columns):
        data['INCOME_PER_CHILD'] = data['AMT_INCOME_TOTAL'] / (data['CNT_CHILDREN'] + 1)

    # Reachability score
    flags = [f for f in ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'] if f in data.columns]
    if flags:
        data['CONTACT_SCORE'] = data[flags].sum(axis=1)

    # ── Synthetic credit-limit target (regression) ──────────────────────────
    # In a production setting this would come from actual bank records.
    # Here we derive a realistic proxy:
    #   base   = 20 % of annual income
    #   bonus  = employment stability multiplier
    #   cap    = $100 k   floor = $500
    if 'AMT_INCOME_TOTAL' in data.columns:
        np.random.seed(42)
        noise  = np.random.normal(1.0, 0.08, len(data))          # ±8 % noise
        emp_m  = np.where(data.get('IS_EMPLOYED', 0) == 1, 1.15, 0.80)
        edu_m  = data.get('NAME_EDUCATION_TYPE', pd.Series(['Secondary / secondary special'] * len(data))).map({
            'Higher education':               1.20,
            'Academic degree':                1.30,
            'Incomplete higher':              1.05,
            'Secondary / secondary special':  1.00,
            'Lower secondary':                0.85
        }).fillna(1.0).values
        data['CREDIT_LIMIT'] = (
            data['AMT_INCOME_TOTAL'] * 0.20 * emp_m * edu_m * noise
        ).clip(500, 100_000).round(-2)      # round to nearest $100

    log.info(f"Features after engineering: {list(data.columns)}")
    return data


def encode_and_split(data: pd.DataFrame):
    """One-hot encode categoricals, split into X/y for classifier and regressor."""
    log.info("Encoding categorical variables …")

    # Separate targets before encoding
    y_cls = data['target'].copy()
    y_reg = data['CREDIT_LIMIT'].copy()

    X = data.drop(columns=['target', 'CREDIT_LIMIT'])
    X = pd.get_dummies(X, drop_first=True)

    log.info(f"Feature matrix shape: {X.shape}")
    return X, y_cls, y_reg


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION TRAINING
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIERS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, min_samples_split=10,
        random_state=42, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=10,
        random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42
    ),
}

# Hyperparameter grid for the final best model (Random Forest / GB)
TUNING_GRID = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [8, 12],
        'min_samples_split': [5, 10],
    },
    'Gradient Boosting': {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
    },
}


def train_classifiers(X_train, X_test, y_train, y_test):
    log.info("\n" + "="*60)
    log.info("CLASSIFICATION — TRAINING MULTIPLE MODELS")
    log.info("="*60)

    results, trained = [], {}

    for name, clf in CLASSIFIERS.items():
        log.info(f"\nTraining {name} …")
        clf.fit(X_train, y_train)
        trained[name] = clf

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Overfitting check: train F1 vs test F1
        train_f1 = f1_score(y_train, clf.predict(X_train))
        test_f1  = f1_score(y_test, y_pred)
        gap = train_f1 - test_f1

        # Find best probability threshold via F1
        best_thresh, best_f1 = 0.5, test_f1
        for t in np.arange(0.3, 0.75, 0.05):
            f = f1_score(y_test, (y_prob >= t).astype(int))
            if f > best_f1:
                best_f1, best_thresh = f, t

        results.append({
            'Model': name,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall':    recall_score(y_test, y_pred),
            'F1':        test_f1,
            'Train_F1':  train_f1,
            'Overfit_Gap': gap,
            'Best_Threshold': round(best_thresh, 2),
        })

        log.info(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        log.info(f"  F1       : {test_f1:.4f}  (train F1: {train_f1:.4f},  gap: {gap:.4f})")
        log.info(f"  Threshold: {best_thresh:.2f}")
        if gap > 0.10:
            log.warning(f"  ⚠️  Overfitting detected for {name} (gap={gap:.3f})")

    return results, trained


def tune_best_model(best_name, trained, X_train, y_train):
    """Run GridSearchCV on the best-performing model if a grid is defined."""
    if best_name not in TUNING_GRID:
        log.info(f"No tuning grid for {best_name}; using default.")
        return trained[best_name]

    log.info(f"\nHyperparameter tuning for {best_name} …")
    grid = GridSearchCV(
        trained[best_name],
        TUNING_GRID[best_name],
        cv=5, scoring='f1', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    log.info(f"  Best params : {grid.best_params_}")
    log.info(f"  Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION TRAINING (Credit Limit)
# ─────────────────────────────────────────────────────────────────────────────

def train_regressor(X_train, X_test, y_train, y_test):
    log.info("\n" + "="*60)
    log.info("REGRESSION — CREDIT LIMIT PREDICTION")
    log.info("="*60)

    regressors = {
        'Ridge Regression':      Ridge(alpha=10.0),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    best_reg, best_r2, best_name = None, -np.inf, ''

    for name, reg in regressors.items():
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        log.info(f"  {name:30s}  MAE={mae:,.0f}  R²={r2:.4f}")

        if r2 > best_r2:
            best_r2, best_reg, best_name = r2, reg, name

    log.info(f"\n  ✅ Best regressor: {best_name}  (R²={best_r2:.4f})")
    return best_reg


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def log_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        top = imp.nlargest(top_n)
        log.info("\nTop feature importances (classifier):")
        for feat, val in top.items():
            log.info(f"  {feat:<45s} {val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(clf, reg, scaler, feature_columns, threshold, out_dir='.'):
    """Persist all model artifacts to disk."""
    os.makedirs(out_dir, exist_ok=True)
    artifacts = {
        'best_model.pkl':      clf,
        'credit_limit_model.pkl': reg,
        'scaler.pkl':          scaler,
        'feature_columns.pkl': feature_columns,
        'threshold.pkl':       threshold,
    }
    for fname, obj in artifacts.items():
        path = os.path.join(out_dir, fname)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    log.info(f"\nAll artifacts saved to '{out_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("AprovAi — MODEL TRAINING PIPELINE")
    log.info("=" * 60)

    # 1. Load
    application, credit = load_data()

    # 2. Target creation
    credit_agg = create_target_variable(credit)

    # 3. Merge & clean
    data = merge_and_clean(application, credit_agg)

    # 4. Feature engineering (also builds CREDIT_LIMIT column)
    data = feature_engineering(data)

    # 5. Encode & extract targets
    X, y_cls, y_reg = encode_and_split(data)
    feature_columns = X.columns.tolist()

    # 6. Train/test split (stratified on approval target)
    X_tr, X_te, yc_tr, yc_te = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    _, _, yr_tr, yr_te = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # 7. Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 8. Train classifiers
    results, trained_clfs = train_classifiers(X_tr_s, X_te_s, yc_tr, yc_te)

    # 9. Select best classifier
    df_res = pd.DataFrame(results)
    log.info("\n" + df_res.to_string(index=False))
    best_idx   = df_res['F1'].idxmax()
    best_name  = df_res.loc[best_idx, 'Model']
    best_thresh= float(df_res.loc[best_idx, 'Best_Threshold'])
    log.info(f"\nBest classifier: {best_name}  F1={df_res.loc[best_idx,'F1']:.4f}")

    # 10. Tune best classifier
    best_clf = tune_best_model(best_name, trained_clfs, X_tr_s, yc_tr)

    # 11. Feature importance
    log_feature_importance(best_clf, feature_columns)

    # 12. Train regressor
    best_reg = train_regressor(X_tr_s, X_te_s, yr_tr, yr_te)

    # 13. Save
    save_artifacts(best_clf, best_reg, scaler, feature_columns, best_thresh)

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE ✅")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
