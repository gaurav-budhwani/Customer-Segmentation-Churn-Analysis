"""
Machine learning models for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
import joblib
import os

class ChurnPredictor:
    """Class for building and evaluating churn prediction models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for machine learning models.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Define feature categories
        categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                               'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                               'PaperlessBilling', 'PaymentMethod']
        
        numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Create working copy
        df_model = df.copy()
        
        # Encode categorical variables
        for col in categorical_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df_model[col] = self.encoders[col].fit_transform(df_model[col].astype(str))
            else:
                df_model[col] = self.encoders[col].transform(df_model[col].astype(str))
        
        # Prepare feature matrix and target
        self.feature_names = categorical_features + numerical_features
        X = df_model[self.feature_names]
        y = df_model['Churn']
        
        return X, y, self.feature_names
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """
        Train multiple churn prediction models.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features for logistic regression
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        X_train_scaled[numerical_cols] = self.scalers['standard'].fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scalers['standard'].transform(X_test[numerical_cols])
        
        results = {}
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        
        lr_pred = lr_model.predict(X_test_scaled)
        lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models['logistic_regression'] = lr_model
        results['logistic_regression'] = {
            'accuracy': lr_model.score(X_test_scaled, y_test),
            'auc_roc': roc_auc_score(y_test, lr_pred_proba),
            'predictions': lr_pred,
            'probabilities': lr_pred_proba,
            'classification_report': classification_report(y_test, lr_pred, output_dict=True)
        }
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        self.models['random_forest'] = rf_model
        results['random_forest'] = {
            'accuracy': rf_model.score(X_test, y_test),
            'auc_roc': roc_auc_score(y_test, rf_pred_proba),
            'predictions': rf_pred,
            'probabilities': rf_pred_proba,
            'feature_importance': dict(zip(self.feature_names, rf_model.feature_importances_)),
            'classification_report': classification_report(y_test, rf_pred, output_dict=True)
        }
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        return results
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of the model to get importance from
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train models first.")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError(f"Model {model_name} does not have feature_importances_ attribute")
    
    def predict_churn_probability(self, customer_data: pd.DataFrame, model_name: str = 'random_forest') -> np.ndarray:
        """
        Predict churn probability for new customers.
        
        Args:
            customer_data: DataFrame with customer features
            model_name: Name of the model to use for prediction
            
        Returns:
            Array of churn probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train models first.")
        
        # Prepare features
        X_pred, _, _ = self.prepare_features(customer_data)
        
        # Scale if needed
        if model_name == 'logistic_regression':
            numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            X_pred[numerical_cols] = self.scalers['standard'].transform(X_pred[numerical_cols])
        
        # Predict
        probabilities = self.models[model_name].predict_proba(X_pred)[:, 1]
        
        return probabilities
    
    def save_models(self, save_dir: str = 'results/models/'):
        """
        Save trained models and preprocessors.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(save_dir, f'{name}_model.pkl'))
        
        # Save scalers and encoders
        joblib.dump(self.scalers, os.path.join(save_dir, 'scalers.pkl'))
        joblib.dump(self.encoders, os.path.join(save_dir, 'encoders.pkl'))
        joblib.dump(self.feature_names, os.path.join(save_dir, 'feature_names.pkl'))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str = 'results/models/'):
        """
        Load trained models and preprocessors.
        
        Args:
            save_dir: Directory to load models from
        """
        # Load models
        for model_file in os.listdir(save_dir):
            if model_file.endswith('_model.pkl'):
                model_name = model_file.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(os.path.join(save_dir, model_file))
        
        # Load scalers and encoders
        self.scalers = joblib.load(os.path.join(save_dir, 'scalers.pkl'))
        self.encoders = joblib.load(os.path.join(save_dir, 'encoders.pkl'))
        self.feature_names = joblib.load(os.path.join(save_dir, 'feature_names.pkl'))
        
        print(f"Models loaded from {save_dir}")
