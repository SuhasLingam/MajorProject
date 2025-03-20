from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os
from typing import Dict, List
from sklearn.metrics import roc_auc_score

class BaseModels:
    def __init__(self, random_state: int = 42):
        """
        Initialize base models with optimized parameters
        Args:
            random_state (int): Random seed
        """
        self.random_state = random_state
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_child_weight=1,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=4,  # Handle class imbalance
                random_state=random_state,
                n_jobs=-1
            )
        }
        self.feature_importances_ = {}
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train all base models and compute feature importance
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training targets
        """
        print("\nTraining base models with feature importance analysis...")
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X, y)
            
            # Store feature importances
            if hasattr(model, 'feature_importances_'):
                self.feature_importances_[name] = model.feature_importances_
            
            # Calculate and print AUC-ROC score
            y_pred_proba = model.predict_proba(X)[:, 1]
            auc_score = roc_auc_score(y, y_pred_proba)
            print(f"{name} AUC-ROC score: {auc_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from all base models
        Args:
            X (np.ndarray): Input features
        Returns:
            np.ndarray: Stacked predictions
        """
        predictions = []
        for model in self.models.values():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        return np.column_stack(predictions)
    
    def save_models(self, output_dir: str):
        """
        Save all base models and feature importances
        Args:
            output_dir (str): Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f'{name}.pkl'))
        
        # Save feature importances
        if self.feature_importances_:
            np.save(os.path.join(output_dir, 'feature_importances.npy'), 
                   self.feature_importances_)

class MetaModel:
    def __init__(self, random_state: int = 42):
        """
        Initialize meta model with optimized parameters
        Args:
            random_state (int): Random seed
        """
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train meta model
        Args:
            X (np.ndarray): Base model predictions
            y (np.ndarray): True labels
        """
        print("\nTraining meta model...")
        self.model.fit(X, y)
        
        # Calculate and print AUC-ROC score
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y, y_pred_proba)
        print(f"Meta model AUC-ROC score: {auc_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from meta model
        Args:
            X (np.ndarray): Base model predictions
        Returns:
            np.ndarray: Final predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions from meta model
        Args:
            X (np.ndarray): Base model predictions
        Returns:
            np.ndarray: Probability predictions
        """
        return self.model.predict_proba(X)
    
    def save_model(self, output_dir: str):
        """
        Save meta model and coefficients
        Args:
            output_dir (str): Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(output_dir, 'meta_model.pkl'))
        
        # Save model coefficients
        if hasattr(self.model, 'coef_'):
            np.save(os.path.join(output_dir, 'meta_model_coefficients.npy'), 
                   self.model.coef_) 