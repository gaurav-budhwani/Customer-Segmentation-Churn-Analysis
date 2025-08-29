"""
Script to download the Kaggle Telco Customer Churn Dataset.
"""

import os
import requests
import zipfile
from pathlib import Path

def download_sample_data():
    """
    Create a sample dataset if Kaggle data is not available.
    This creates a realistic sample for testing the analysis pipeline.
    """
    import pandas as pd
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_customers = 7043  # Similar to actual Kaggle dataset size
    
    # Customer demographics
    customer_ids = [f"C{i:04d}" for i in range(1, n_customers + 1)]
    genders = np.random.choice(['Male', 'Female'], n_customers)
    senior_citizens = np.random.choice([0, 1], n_customers, p=[0.84, 0.16])
    partners = np.random.choice(['Yes', 'No'], n_customers, p=[0.52, 0.48])
    dependents = np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70])
    
    # Tenure (months with company)
    tenure = np.random.exponential(scale=20, size=n_customers).astype(int)
    tenure = np.clip(tenure, 0, 72)
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], n_customers, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == 'Yes',
        np.random.choice(['Yes', 'No'], n_customers, p=[0.42, 0.58]),
        'No phone service'
    )
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22])
    
    # Additional services (dependent on internet)
    def internet_dependent_service(internet_svc, yes_prob=0.3):
        return np.where(
            internet_svc == 'No',
            'No internet service',
            np.random.choice(['Yes', 'No'], len(internet_svc), p=[yes_prob, 1-yes_prob])
        )
    
    online_security = internet_dependent_service(internet_service, 0.28)
    online_backup = internet_dependent_service(internet_service, 0.34)
    device_protection = internet_dependent_service(internet_service, 0.34)
    tech_support = internet_dependent_service(internet_service, 0.29)
    streaming_tv = internet_dependent_service(internet_service, 0.38)
    streaming_movies = internet_dependent_service(internet_service, 0.39)
    
    # Contract and billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_customers, p=[0.34, 0.19, 0.22, 0.25])
    
    # Charges (influenced by services and tenure)
    base_charge = 20
    service_charges = (
        (phone_service == 'Yes') * 10 +
        (internet_service == 'DSL') * 25 +
        (internet_service == 'Fiber optic') * 45 +
        (multiple_lines == 'Yes') * 5 +
        (streaming_tv == 'Yes') * 10 +
        (streaming_movies == 'Yes') * 10 +
        (online_security == 'Yes') * 5 +
        (tech_support == 'Yes') * 5
    )
    
    monthly_charges = base_charge + service_charges + np.random.normal(0, 5, n_customers)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    total_charges = monthly_charges * tenure + np.random.normal(0, 50, n_customers)
    total_charges = np.maximum(total_charges, 0)
    
    # Churn (influenced by various factors)
    churn_prob = (
        0.1 +  # Base churn rate
        (contract == 'Month-to-month') * 0.3 +
        (payment_method == 'Electronic check') * 0.15 +
        (senior_citizens == 1) * 0.1 +
        (tenure < 12) * 0.2 +
        (monthly_charges > 80) * 0.1 +
        (internet_service == 'Fiber optic') * 0.05 -
        (partners == 'Yes') * 0.05 -
        (dependents == 'Yes') * 0.05 -
        (online_security == 'Yes') * 0.05
    )
    
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(1, churn_prob)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior_citizens,
        'Partner': partners,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'Churn': np.where(churn == 1, 'Yes', 'No')
    })
    
    return sample_data

def main():
    """Main function to handle data acquisition."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if Kaggle dataset exists
    kaggle_file = data_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    if not kaggle_file.exists():
        print("Kaggle dataset not found. Creating sample dataset for demonstration...")
        sample_data = download_sample_data()
        sample_file = data_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        sample_data.to_csv(sample_file, index=False)
        print(f"Sample dataset created at {sample_file}")
        print(f"Dataset shape: {sample_data.shape}")
        print(f"Churn rate: {(sample_data['Churn'] == 'Yes').mean():.2%}")
    else:
        print(f"Dataset already exists at {kaggle_file}")

if __name__ == "__main__":
    main()
