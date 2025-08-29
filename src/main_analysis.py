"""
Main script to run the complete customer churn analysis pipeline.
"""

import os
import pandas as pd
import numpy as np
from data_loader import ChurnDataLoader
from churn_models import ChurnPredictor
from customer_segmentation import CustomerSegmentation
import matplotlib.pyplot as plt
import seaborn as sns

def run_complete_analysis():
    """Run the complete customer churn analysis pipeline."""
    
    print("=" * 60)
    print("CUSTOMER CHURN ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    loader = ChurnDataLoader()
    
    # Try to load data, create sample if not available
    df_raw = loader.load_telco_data()
    if df_raw is None:
        print("Creating sample dataset...")
        from download_data import download_sample_data
        df_raw = download_sample_data()
        
    df = loader.preprocess_data(df_raw)
    summary = loader.get_data_summary(df)
    
    print(f"Dataset loaded: {summary['total_customers']} customers")
    print(f"Churn rate: {summary['churn_rate']:.2%}")
    
    # Create SQLite database
    db_path = loader.create_sqlite_db(df)
    
    # Step 2: Exploratory Data Analysis
    print("\n2. Performing exploratory data analysis...")
    
    # Spending segment analysis
    df['SpendingSegment'] = pd.cut(df['MonthlyCharges'], 
                                  bins=[0, 35, 65, float('inf')], 
                                  labels=['Low Spender', 'Medium Spender', 'High Spender'])
    
    spending_analysis = df.groupby('SpendingSegment')['Churn'].agg(['count', 'sum', 'mean'])
    spending_analysis.columns = ['TotalCustomers', 'ChurnedCustomers', 'ChurnRate']
    spending_analysis['ChurnRate'] = spending_analysis['ChurnRate'] * 100
    
    print("\nChurn Analysis by Spending Segment:")
    print(spending_analysis)
    
    # Contract analysis
    contract_analysis = df.groupby('Contract')['Churn'].agg(['count', 'sum', 'mean'])
    contract_analysis.columns = ['TotalCustomers', 'ChurnedCustomers', 'ChurnRate']
    contract_analysis['ChurnRate'] = contract_analysis['ChurnRate'] * 100
    
    print("\nChurn Analysis by Contract Type:")
    print(contract_analysis)
    
    # Step 3: Machine Learning Models
    print("\n3. Training churn prediction models...")
    predictor = ChurnPredictor()
    X, y, feature_names = predictor.prepare_features(df)
    results = predictor.train_models(X, y)
    
    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"{model_name.title()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    
    # Feature importance
    feature_importance = predictor.get_feature_importance()
    print(f"\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    # Step 4: Customer Segmentation
    print("\n4. Performing customer segmentation...")
    segmenter = CustomerSegmentation()
    df_cluster_ready = segmenter.prepare_clustering_data(df)
    
    # Find optimal clusters
    cluster_metrics = segmenter.find_optimal_clusters(df_cluster_ready)
    optimal_k = 4  # Based on typical business segments
    
    # Perform clustering
    df_clustered = segmenter.perform_clustering(df_cluster_ready, n_clusters=optimal_k)
    cluster_analysis = segmenter.analyze_clusters(df_clustered)
    
    print(f"\nCustomer Segmentation Results ({optimal_k} clusters):")
    print(cluster_analysis[['CustomerCount', 'ChurnRate', 'AvgTenure', 'AvgMonthlyCharges']])
    
    # Step 5: Export results
    print("\n5. Exporting results...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Export model results
    predictor.save_models()
    
    # Export segmentation results
    segmenter.export_cluster_results(df_clustered, cluster_analysis)
    
    # Export summary reports
    spending_analysis.to_csv('results/spending_segment_analysis.csv')
    contract_analysis.to_csv('results/contract_analysis.csv')
    feature_importance.to_csv('results/feature_importance.csv', index=False)
    
    # Create high-risk customer list using original dataframe with cluster info
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = df_clustered['Cluster']

    high_risk_customers = df_with_clusters[
        (df_with_clusters['Churn'] == 0) &  # Current customers only
        (
            (df_with_clusters['Contract'] == 'Month-to-month') |
            (df_with_clusters['tenure'] < 12) |
            (df_with_clusters['PaymentMethod'] == 'Electronic check')
        )
    ].copy()

    # Calculate risk score
    high_risk_customers['RiskScore'] = (
        (high_risk_customers['Contract'] == 'Month-to-month').astype(int) +
        (high_risk_customers['tenure'] < 12).astype(int) +
        (high_risk_customers['PaymentMethod'] == 'Electronic check').astype(int) +
        (high_risk_customers['MonthlyCharges'] > 70).astype(int)
    )
    
    # Get churn probabilities
    churn_probabilities = predictor.predict_churn_probability(high_risk_customers)
    high_risk_customers['ChurnProbability'] = churn_probabilities
    
    # Sort by risk
    high_risk_customers = high_risk_customers.sort_values(['RiskScore', 'ChurnProbability'], ascending=False)
    
    # Export top 500 high-risk customers
    high_risk_export = high_risk_customers[['customerID', 'tenure', 'Contract', 'PaymentMethod', 
                                           'MonthlyCharges', 'TotalCharges', 'Cluster', 
                                           'RiskScore', 'ChurnProbability']].head(500)
    high_risk_export.to_csv('results/high_risk_customers.csv', index=False)
    
    print(f"High-risk customers identified: {len(high_risk_customers)}")
    print(f"Top 500 exported to results/high_risk_customers.csv")
    
    # Step 6: Generate insights summary
    print("\n6. Generating insights summary...")
    
    insights = {
        'total_customers': len(df),
        'overall_churn_rate': df['Churn'].mean(),
        'highest_risk_segment': cluster_analysis.loc[cluster_analysis['ChurnRate'].idxmax(), 'ClusterLabel'],
        'highest_churn_contract': contract_analysis.loc[contract_analysis['ChurnRate'].idxmax()].name,
        'revenue_at_risk': df[df['Churn'] == 1]['MonthlyCharges'].sum(),
        'top_churn_factors': feature_importance.head(3)['feature'].tolist()
    }
    
    print("\nKEY INSIGHTS:")
    print(f"• Overall churn rate: {insights['overall_churn_rate']:.1%}")
    print(f"• Highest risk segment: {insights['highest_risk_segment']}")
    print(f"• Highest churn contract type: {insights['highest_churn_contract']}")
    print(f"• Monthly revenue at risk: ${insights['revenue_at_risk']:,.2f}")
    print(f"• Top churn factors: {', '.join(insights['top_churn_factors'])}")
    
    # Save insights
    with open('results/key_insights.txt', 'w') as f:
        f.write("CUSTOMER CHURN ANALYSIS - KEY INSIGHTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall churn rate: {insights['overall_churn_rate']:.1%}\n")
        f.write(f"Highest risk segment: {insights['highest_risk_segment']}\n")
        f.write(f"Highest churn contract type: {insights['highest_churn_contract']}\n")
        f.write(f"Monthly revenue at risk: ${insights['revenue_at_risk']:,.2f}\n")
        f.write(f"Top churn factors: {', '.join(insights['top_churn_factors'])}\n")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nFiles created:")
    print("• SQL queries: sql/")
    print("• Python analysis: notebooks/churn_analysis.ipynb")
    print("• Model results: results/")
    print("• Power BI setup: powerbi/dashboard_setup.md")
    print("\nNext steps:")
    print("1. Run: python src/download_data.py (to get sample data)")
    print("2. Run: python src/main_analysis.py (this script)")
    print("3. Open Power BI and follow powerbi/dashboard_setup.md")
    print("4. Import results/*.csv files into Power BI")

if __name__ == "__main__":
    run_complete_analysis()
