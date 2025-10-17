"""
Machine Learning model definitions and factory
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict


class ModelFactory:
    """Factory for creating ML models"""
    
    @staticmethod
    def get_models(params: Dict = None) -> Dict:
        """
        Get dictionary of ML models
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Dictionary of model name -> model instance
        """
        if params is None:
            params = {}
        
        models = {
            "Logistic Regression": LogisticRegression(
                **params.get('logistic_regression', {
                    'max_iter': 1000,
                    'class_weight': 'balanced',
                    'random_state': 42
                })
            ),
            "Random Forest": RandomForestClassifier(
                **params.get('random_forest', {
                    'n_estimators': 100,
                    'class_weight': 'balanced',
                    'random_state': 42
                })
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                **params.get('gradient_boosting', {
                    'n_estimators': 100,
                    'random_state': 42
                })
            ),
            "SVM": SVC(
                **params.get('svm', {
                    'class_weight': 'balanced',
                    'random_state': 42
                })
            ),
            "KNN": KNeighborsClassifier(
                **params.get('knn', {
                    'n_neighbors': 5
                })
            ),
            "Decision Tree": DecisionTreeClassifier(
                **params.get('decision_tree', {
                    'class_weight': 'balanced',
                    'random_state': 42
                })
            ),
            "XGBoost": XGBClassifier(
                **params.get('xgboost', {
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss',
                    'random_state': 42
                })
            ),
            "LightGBM": LGBMClassifier(
                **params.get('lightgbm', {
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'verbose': -1
                })
            )
        }
        
        return models
    
    @staticmethod
    def get_model(model_name: str, params: Dict = None):
        """
        Get a specific model by name
        
        Args:
            model_name: Name of the model
            params: Model parameters
            
        Returns:
            Model instance
        """
        models = ModelFactory.get_models(params)
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
        return models[model_name]