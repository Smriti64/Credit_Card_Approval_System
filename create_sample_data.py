# create_sample_data.py
import pandas as pd
import numpy as np

# Create sample application_record.csv
np.random.seed(42)
n_samples = 1000

application_data = {
    'ID': range(1, n_samples + 1),
    'CODE_GENDER': np.random.choice(['M', 'F'], n_samples),
    'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
    'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_samples),
    'CNT_CHILDREN': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'AMT_INCOME_TOTAL': np.random.uniform(20000, 200000, n_samples),
    'NAME_INCOME_TYPE': np.random.choice(
        ['Working', 'Commercial associate', 'Pensioner', 'State servant'],
        n_samples, p=[0.5, 0.25, 0.15, 0.1]
    ),
    'NAME_EDUCATION_TYPE': np.random.choice(
        ['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary'],
        n_samples, p=[0.3, 0.4, 0.2, 0.1]
    ),
    'NAME_FAMILY_STATUS': np.random.choice(
        ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
        n_samples, p=[0.5, 0.25, 0.1, 0.1, 0.05]
    ),
    'NAME_HOUSING_TYPE': np.random.choice(
        ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment'],
        n_samples, p=[0.6, 0.2, 0.1, 0.1]
    ),
    'DAYS_BIRTH': np.random.randint(-25000, -7000, n_samples),
    'DAYS_EMPLOYED': np.random.choice(
        list(range(-5000, 0)) + [365243],
        n_samples
    ),
    'FLAG_MOBIL': np.ones(n_samples, dtype=int),
    'FLAG_WORK_PHONE': np.random.choice([0, 1], n_samples),
    'FLAG_PHONE': np.random.choice([0, 1], n_samples),
    'FLAG_EMAIL': np.random.choice([0, 1], n_samples),
    'OCCUPATION_TYPE': np.random.choice(
        ['Laborers', 'Core staff', 'Managers', 'Drivers', 'Sales staff', None],
        n_samples, p=[0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
    ),
    'CNT_FAM_MEMBERS': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
}

application_df = pd.DataFrame(application_data)
application_df.to_csv('data/application_record.csv', index=False)

# Create sample credit_record.csv
credit_records = []
for id in range(1, n_samples + 1):
    n_months = np.random.randint(3, 24)
    for month in range(-n_months, 0):
        status = np.random.choice(['C', 'X', '0', '1', '2'], p=[0.5, 0.3, 0.1, 0.05, 0.05])
        credit_records.append({
            'ID': id,
            'MONTHS_BALANCE': month,
            'STATUS': status
        })

credit_df = pd.DataFrame(credit_records)
credit_df.to_csv('data/credit_record.csv', index=False)

print("Sample data created successfully!")
