"""
Data loading and preprocessing utilities for customer churn analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sqlite3
import os

class ChurnDataLoader:
    """Class to handle loading and preprocessing of customer churn data."""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        
    def load_telco_data(self, filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
        """
        Load the Telco Customer Churn dataset from CSV.

        Args:
            filename: Name of the CSV file

        Returns:
            DataFrame with the loaded data
        """
        # Try multiple possible paths (for notebook vs script execution)
        possible_paths = [
            os.path.join(self.data_path, filename),
            os.path.join("../data", filename),  # From notebooks directory
            os.path.join("data", filename),     # From root directory
            filename  # Direct path
        ]

        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if file_path is None:
            print(f"Data file not found. Tried paths:")
            for path in possible_paths:
                print(f"  - {path}")
            print("Please download the Kaggle Telco Customer Churn Dataset and place it in the data/ directory")
            return None

        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the customer churn data.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Preprocessed dataframe
        """
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle TotalCharges column (convert to numeric)
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

        # Fill missing TotalCharges with 0 (likely new customers)
        df_processed.loc[:, 'TotalCharges'] = df_processed['TotalCharges'].fillna(0)
        
        # Convert binary categorical variables to numeric
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        for col in binary_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
        
        # Convert SeniorCitizen to consistent format
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)
        
        # Create additional features
        df_processed['AvgMonthlySpend'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 1)
        df_processed['HasMultipleServices'] = (
            (df_processed['PhoneService'] == 1) & 
            (df_processed['InternetService'] != 'No')
        ).astype(int)
        
        return df_processed
    
    def create_sqlite_db(self, df: pd.DataFrame, db_name: str = "churn_analysis.db") -> str:
        """
        Create SQLite database from the dataframe.
        
        Args:
            df: Preprocessed dataframe
            db_name: Name of the SQLite database file
            
        Returns:
            Path to the created database
        """
        db_path = os.path.join(self.data_path, db_name)
        
        # Create connection and save data
        conn = sqlite3.connect(db_path)
        df.to_sql('customers', conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"SQLite database created at {db_path}")
        return db_path
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            df: Dataframe to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_customers': len(df),
            'churned_customers': df['Churn'].sum() if 'Churn' in df.columns else None,
            'churn_rate': df['Churn'].mean() if 'Churn' in df.columns else None,
            'avg_tenure': df['tenure'].mean(),
            'avg_monthly_charges': df['MonthlyCharges'].mean(),
            'avg_total_charges': df['TotalCharges'].mean(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return summary
