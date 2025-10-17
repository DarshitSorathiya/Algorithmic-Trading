"""
Machine Learning training and evaluation module
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Tuple, List
from .models import ModelFactory


class MLTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self):
        """Initialize MLTrainer"""
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        self.results = []
        self.confusion_matrices = {}
    
    def prepare_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Tuple:
        """
        Prepare data for ML training
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            
        Returns:
            Tuple of encoded data
        """
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if X_test is not None and y_test is not None:
            y_test_encoded = self.label_encoder.transform(y_test)
            return X_train, y_train_encoded, X_test, y_test_encoded
        
        return X_train, y_train_encoded
    
    def calculate_class_weights(self, y_train_encoded: np.ndarray) -> Dict:
        """
        Calculate class weights for imbalanced data
        
        Args:
            y_train_encoded: Encoded training labels
            
        Returns:
            Dictionary of class weights
        """
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_encoded),
            y=y_train_encoded
        )
        return dict(zip(np.unique(y_train_encoded), class_weights))
    
    def train_models(self, X_train: pd.DataFrame, y_train_encoded: np.ndarray,
                    X_test: pd.DataFrame, y_test_encoded: np.ndarray,
                    model_params: Dict = None) -> pd.DataFrame:
        """
        Train multiple models and evaluate
        
        Args:
            X_train: Training features
            y_train_encoded: Encoded training labels
            X_test: Test features
            y_test_encoded: Encoded test labels
            model_params: Dictionary of model parameters
            
        Returns:
            DataFrame with results
        """
        models = ModelFactory.get_models(model_params)
        
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        best_f1 = -1
        best_model_name = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with scaling
            pipe = make_pipeline(StandardScaler(), model)
            
            # Train
            pipe.fit(X_train, y_train_encoded)
            
            # Predict
            y_pred_test = pipe.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test_encoded, y_pred_test)
            prec = precision_score(y_test_encoded, y_pred_test, 
                                  average='weighted', zero_division=0)
            rec = recall_score(y_test_encoded, y_pred_test, 
                              average='weighted', zero_division=0)
            f1 = f1_score(y_test_encoded, y_pred_test, 
                         average='weighted', zero_division=0)
            
            # Store results
            self.results.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1
            })
            
            # Store confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred_test)
            self.confusion_matrices[name] = cm
            
            # Store trained model
            self.trained_models[name] = {
                'pipeline': pipe,
                'predictions': y_pred_test,
                'f1_score': f1
            }
            
            # Track best model
            if f1 > best_f1:
                best_model_name = name
                best_f1 = f1
        
        results_df = pd.DataFrame(self.results).sort_values(
            by='F1 Score', ascending=False
        )
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(results_df.to_string(index=False))
        
        return results_df, best_model_name
    
    def get_best_model(self) -> Tuple[str, object, np.ndarray]:
        """
        Get the best performing model
        
        Returns:
            Tuple of (model_name, pipeline, predictions)
        """
        if not self.trained_models:
            raise ValueError("No models trained yet. Call train_models first.")
        
        best_name = max(self.trained_models.items(), 
                       key=lambda x: x[1]['f1_score'])[0]
        best_info = self.trained_models[best_name]
        
        return best_name, best_info['pipeline'], best_info['predictions']
    
    def get_classification_report(self, y_test_encoded: np.ndarray, 
                                 model_name: str = None) -> str:
        """
        Get classification report for a model
        
        Args:
            y_test_encoded: True test labels
            model_name: Name of model (None for best model)
            
        Returns:
            Classification report string
        """
        if model_name is None:
            model_name, _, y_pred = self.get_best_model()
        else:
            y_pred = self.trained_models[model_name]['predictions']
        
        unique_classes = np.unique(np.concatenate([y_test_encoded, y_pred]))
        class_names_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
        present_class_names = [class_names_map[c] for c in unique_classes]
        
        return classification_report(
            y_test_encoded, 
            y_pred,
            labels=unique_classes,
            target_names=present_class_names,
            zero_division=0
        )
    
    def get_feature_importance(self, feature_names: List[str], 
                              model_name: str = None, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Args:
            feature_names: List of feature names
            model_name: Name of model (None for best model)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if model_name is None:
            model_name, pipe, _ = self.get_best_model()
        else:
            pipe = self.trained_models[model_name]['pipeline']
        
        # Get the actual model from pipeline
        model = pipe.named_steps[list(pipe.named_steps.keys())[1]]
        
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        }).sort_values('Importance', ascending=False)
        
        return importance_df