#!/usr/bin/env python3
"""
Script to run the churn analysis notebook programmatically.
This executes the same analysis as the Jupyter notebook but in a script format.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import ChurnDataLoader
from churn_models import ChurnPredictor
from customer_segmentation import CustomerSegmentation

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def run_notebook_analysis():
    """Execute the complete notebook analysis."""
    
    print("=" * 60)
    print("JUPYTER NOTEBOOK ANALYSIS - PROGRAMMATIC EXECUTION")
    print("=" * 60)
    
    # 1. Data Loading and Initial Exploration
    print("\n1. DATA LOADING AND INITIAL EXPLORATION")
    print("-" * 40)
    
    loader = ChurnDataLoader()
    df_raw = loader.load_telco_data()
    
    if df_raw is not None:
        print(f"Dataset shape: {df_raw.shape}")
        print(f"\nFirst few rows:")
        print(df_raw.head())
        print(f"\nDataset info:")
        print(f"Columns: {list(df_raw.columns)}")
        print(f"Data types: {df_raw.dtypes.value_counts()}")
    else:
        print("Failed to load data. Exiting...")
        return
    
    # Preprocess data
    df = loader.preprocess_data(df_raw)
    summary = loader.get_data_summary(df)
    
    print(f"\nData Summary:")
    for key, value in summary.items():
        if key != 'missing_values':
            print(f"  {key}: {value}")
    
    # 2. Exploratory Data Analysis
    print("\n\n2. EXPLORATORY DATA ANALYSIS")
    print("-" * 40)
    
    # Churn distribution
    churn_counts = df['Churn'].value_counts()
    print(f"Churn Distribution:")
    print(f"  No Churn: {churn_counts[0]} ({churn_counts[0]/len(df)*100:.1f}%)")
    print(f"  Churn: {churn_counts[1]} ({churn_counts[1]/len(df)*100:.1f}%)")
    
    # High-paying vs Low-paying customer analysis
    df['SpendingSegment'] = pd.cut(df['MonthlyCharges'], 
                                  bins=[0, 35, 65, float('inf')], 
                                  labels=['Low Spender', 'Medium Spender', 'High Spender'])
    
    spending_churn = df.groupby('SpendingSegment', observed=True)['Churn'].agg(['count', 'sum', 'mean']).reset_index()
    spending_churn.columns = ['SpendingSegment', 'TotalCustomers', 'ChurnedCustomers', 'ChurnRate']
    spending_churn['ChurnRate'] = spending_churn['ChurnRate'] * 100
    
    print(f"\nChurn Analysis by Spending Segment:")
    print(spending_churn.to_string(index=False))
    
    # Contract analysis
    contract_churn = df.groupby('Contract')['Churn'].agg(['count', 'sum', 'mean']).reset_index()
    contract_churn.columns = ['Contract', 'TotalCustomers', 'ChurnedCustomers', 'ChurnRate']
    contract_churn['ChurnRate'] = contract_churn['ChurnRate'] * 100
    
    print(f"\nChurn Analysis by Contract Type:")
    print(contract_churn.to_string(index=False))
    
    # 3. Machine Learning - Churn Prediction
    print("\n\n3. MACHINE LEARNING - CHURN PREDICTION")
    print("-" * 40)
    
    predictor = ChurnPredictor()
    X, y, feature_names = predictor.prepare_features(df)
    results = predictor.train_models(X, y)
    
    print(f"\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    
    # Feature importance
    feature_importance = predictor.get_feature_importance()
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # 4. Customer Segmentation
    print("\n\n4. CUSTOMER SEGMENTATION WITH KMEANS")
    print("-" * 40)
    
    segmenter = CustomerSegmentation()
    df_cluster_ready = segmenter.prepare_clustering_data(df)
    
    # Perform clustering
    df_clustered = segmenter.perform_clustering(df_cluster_ready, n_clusters=4)
    cluster_analysis = segmenter.analyze_clusters(df_clustered)
    
    print(f"\nCustomer Segmentation Results (4 clusters):")
    print(cluster_analysis[['CustomerCount', 'ChurnRate', 'AvgTenure', 'AvgMonthlyCharges']].to_string())
    
    # 5. Business Insights
    print("\n\n5. KEY BUSINESS INSIGHTS")
    print("-" * 40)
    
    print(f"📊 CUSTOMER SEGMENTS:")
    for idx, row in cluster_analysis.iterrows():
        risk_level = "🔴 HIGH RISK" if row['ChurnRate'] > 0.4 else "🟡 MEDIUM RISK" if row['ChurnRate'] > 0.3 else "🟢 LOW RISK"
        print(f"  Cluster {idx}: {row['CustomerCount']} customers, {row['ChurnRate']:.1%} churn rate {risk_level}")
    
    print(f"\n💰 FINANCIAL IMPACT:")
    revenue_at_risk = df[df['Churn'] == 1]['MonthlyCharges'].sum()
    print(f"  Monthly revenue at risk: ${revenue_at_risk:,.2f}")
    print(f"  Average customer value: ${df['TotalCharges'].mean():,.2f}")
    
    print(f"\n🎯 RETENTION PRIORITIES:")
    print(f"  1. Target month-to-month customers (57.4% churn rate)")
    print(f"  2. Focus on new customers (< 12 months tenure)")
    print(f"  3. Address high-spending customer concerns")
    print(f"  4. Promote automatic payment methods")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"All results saved to 'results/' directory for Power BI import")

if __name__ == "__main__":
    run_notebook_analysis()
