"""
Customer segmentation analysis using KMeans clustering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import os

class CustomerSegmentation:
    """Class for customer segmentation analysis."""
    
    def __init__(self):
        self.kmeans_model = None
        self.scaler = None
        self.feature_names = None
        self.cluster_labels = None
        
    def prepare_clustering_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for clustering analysis.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame ready for clustering
        """
        # Select features for clustering
        clustering_features = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
            'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'
        ]
        
        # Create clustering dataset
        df_cluster = df[clustering_features + ['customerID', 'Churn']].copy()
        
        # Handle missing values
        df_cluster['TotalCharges'] = pd.to_numeric(df_cluster['TotalCharges'], errors='coerce')
        df_cluster.loc[:, 'TotalCharges'] = df_cluster['TotalCharges'].fillna(0)

        # Ensure no NaN values in any clustering features
        for col in clustering_features:
            if df_cluster[col].dtype == 'object':
                df_cluster[col] = df_cluster[col].fillna('Unknown')
            else:
                df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())
        
        # Convert binary categorical to numeric
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df_cluster.columns:
                df_cluster[col] = df_cluster[col].map({'Yes': 1, 'No': 0})
        
        self.feature_names = clustering_features
        return df_cluster
    
    def find_optimal_clusters(self, X: pd.DataFrame, max_k: int = 10) -> dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.

        Args:
            X: Feature matrix for clustering
            max_k: Maximum number of clusters to test

        Returns:
            Dictionary with metrics for different k values
        """
        # Ensure no NaN values before scaling
        X_clean = X[self.feature_names].copy()

        # Check for and handle any remaining NaN values
        if X_clean.isnull().any().any():
            print("Warning: Found NaN values, filling with median/mode...")
            for col in X_clean.columns:
                if X_clean[col].dtype in ['float64', 'int64']:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0] if len(X_clean[col].mode()) > 0 else 0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # Verify no NaN or infinite values after scaling
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("Warning: Found NaN/inf values after scaling, replacing with zeros...")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

        return {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    
    def perform_clustering(self, X: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        """
        Perform KMeans clustering on customer data.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters

        Returns:
            DataFrame with cluster assignments
        """
        # Ensure no NaN values before scaling
        X_clean = X[self.feature_names].copy()

        # Check for and handle any remaining NaN values
        if X_clean.isnull().any().any():
            print("Warning: Found NaN values in clustering data, filling...")
            for col in X_clean.columns:
                if X_clean[col].dtype in ['float64', 'int64']:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0] if len(X_clean[col].mode()) > 0 else 0)

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_clean)
        else:
            X_scaled = self.scaler.transform(X_clean)

        # Verify no NaN or infinite values after scaling
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("Warning: Found NaN/inf values after scaling, replacing...")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

        # Perform clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(X_scaled)

        # Add cluster labels to dataframe
        X_clustered = X.copy()
        X_clustered['Cluster'] = self.cluster_labels

        return X_clustered
    
    def analyze_clusters(self, df_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze characteristics of each cluster.
        
        Args:
            df_clustered: DataFrame with cluster assignments
            
        Returns:
            DataFrame with cluster analysis
        """
        # Group by cluster and calculate statistics
        cluster_analysis = df_clustered.groupby('Cluster').agg({
            'customerID': 'count',
            'Churn': ['sum', 'mean'],
            'tenure': 'mean',
            'MonthlyCharges': 'mean',
            'TotalCharges': 'mean',
            'SeniorCitizen': 'mean',
            'Partner': 'mean',
            'Dependents': 'mean'
        }).round(3)
        
        # Flatten column names
        cluster_analysis.columns = [
            'CustomerCount', 'ChurnedCustomers', 'ChurnRate',
            'AvgTenure', 'AvgMonthlyCharges', 'AvgTotalCharges',
            'SeniorCitizenRate', 'PartnerRate', 'DependentsRate'
        ]
        
        # Add cluster labels
        cluster_analysis['ClusterLabel'] = self.get_cluster_labels(cluster_analysis)
        
        return cluster_analysis
    
    def get_cluster_labels(self, cluster_stats: pd.DataFrame) -> list:
        """
        Generate descriptive labels for clusters based on their characteristics.
        
        Args:
            cluster_stats: DataFrame with cluster statistics
            
        Returns:
            List of cluster labels
        """
        labels = []
        
        for idx, row in cluster_stats.iterrows():
            if row['AvgTenure'] < 20 and row['ChurnRate'] > 0.4:
                labels.append('High-Risk New Customers')
            elif row['AvgTenure'] > 40 and row['ChurnRate'] < 0.2:
                labels.append('Loyal Long-term Customers')
            elif row['AvgMonthlyCharges'] > 70:
                labels.append('High-Value Customers')
            elif row['AvgMonthlyCharges'] < 40:
                labels.append('Budget-Conscious Customers')
            else:
                labels.append(f'Cluster {idx}')
        
        return labels
    
    def visualize_clusters(self, df_clustered: pd.DataFrame):
        """
        Create visualizations for cluster analysis.
        
        Args:
            df_clustered: DataFrame with cluster assignments
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cluster scatter plot: Tenure vs Monthly Charges
        scatter = axes[0,0].scatter(df_clustered['tenure'], df_clustered['MonthlyCharges'], 
                                   c=df_clustered['Cluster'], cmap='viridis', alpha=0.6)
        axes[0,0].set_xlabel('Tenure (months)')
        axes[0,0].set_ylabel('Monthly Charges')
        axes[0,0].set_title('Customer Clusters: Tenure vs Monthly Charges')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # Churn rate by cluster
        cluster_churn = df_clustered.groupby('Cluster')['Churn'].mean()
        axes[0,1].bar(cluster_churn.index, cluster_churn.values)
        axes[0,1].set_xlabel('Cluster')
        axes[0,1].set_ylabel('Churn Rate')
        axes[0,1].set_title('Churn Rate by Cluster')
        
        # Customer count by cluster
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        axes[1,0].bar(cluster_counts.index, cluster_counts.values)
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Number of Customers')
        axes[1,0].set_title('Customer Count by Cluster')
        
        # Average monthly charges by cluster
        cluster_charges = df_clustered.groupby('Cluster')['MonthlyCharges'].mean()
        axes[1,1].bar(cluster_charges.index, cluster_charges.values)
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Average Monthly Charges')
        axes[1,1].set_title('Average Monthly Charges by Cluster')
        
        plt.tight_layout()
        plt.show()
    
    def export_cluster_results(self, df_clustered: pd.DataFrame, cluster_analysis: pd.DataFrame, 
                              output_dir: str = 'results/'):
        """
        Export clustering results for Power BI dashboard.
        
        Args:
            df_clustered: DataFrame with cluster assignments
            cluster_analysis: DataFrame with cluster statistics
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export customer data with clusters
        df_clustered.to_csv(os.path.join(output_dir, 'customers_with_clusters.csv'), index=False)
        
        # Export cluster analysis
        cluster_analysis.to_csv(os.path.join(output_dir, 'cluster_analysis.csv'))
        
        print(f"Cluster results exported to {output_dir}")
