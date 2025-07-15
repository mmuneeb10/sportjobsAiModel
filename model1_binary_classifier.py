#!/usr/bin/env python3
"""
Model 1: Binary Classifier
This model classifies candidates into:
- Class 0: REJECT
- Class 1: Others (SHORTLIST, INTERVIEW, ACCEPT)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, f1_score,
    balanced_accuracy_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model1BinaryClassifier:
    """Binary classifier for REJECT vs Others"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model_name = None
        self.class_weights = None
        
        # Binary mapping
        self.label_mapping = {
            'REJECT': 0,        # Class 0
            'SHORTLIST': 1,     # Class 1
            'INTERVIEW': 1,     # Class 1
            'ACCEPT': 1         # Class 1
        }
        
        # Feature names for interpretability
        self.feature_names = [
            'text_similarity', 'experience_years', 'experience_ratio',
            'meets_experience', 'skill_match_ratio', 'skill_coverage',
            'total_skills', 'matching_skills', 'education_level',
            'certifications', 'industry_keywords', 'communication_score',
            'achievements', 'languages', 'location_match'
        ]
        
        # Initialize multiple models with different strategies
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different models for comparison"""
        
        # Random Forest with balanced class weights
        self.models['rf_balanced'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting with custom parameters
        self.models['gb_classifier'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        
        # Logistic Regression with balanced weights
        self.models['logistic_balanced'] = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            random_state=42,
            solver='liblinear'
        )
        
        # Support Vector Machine
        self.models['svm_balanced'] = SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42,
            gamma='scale'
        )
    
    def convert_labels_to_binary(self, stages: List[str]) -> np.ndarray:
        """Convert stage labels to binary format"""
        return np.array([self.label_mapping[stage] for stage in stages])
    
    def prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features"""
        return self.scaler.fit_transform(X)
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced data"""
        unique_classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
        
        self.class_weights = dict(zip(unique_classes, class_weights))
        logger.info(f"Computed class weights: {self.class_weights}")
        
        # Update models that support sample weights
        if hasattr(self.models['gb_classifier'], 'class_weight'):
            self.models['gb_classifier'].class_weight = self.class_weights
        
        return self.class_weights
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train all models and select the best one"""
        logger.info("Training Model 1 (Binary Classifier)...")
        
        # Compute class weights
        self.compute_class_weights(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Training set distribution: {dict(zip(['REJECT', 'Others'], counts))}")
        
        # Train each model
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\nTraining {model_name}...")
            
            # Train model
            if model_name == 'gb_classifier' and self.class_weights:
                # Gradient Boosting doesn't support class_weight, use sample_weight
                sample_weights = np.array([self.class_weights[y] for y in y_train])
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_proba)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            
            # Store results
            results[model_name] = {
                'f1_score': f1,
                'roc_auc': roc_auc,
                'balanced_accuracy': balanced_acc,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            # Log performance
            logger.info(f"{model_name} - F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, Balanced Acc: {balanced_acc:.4f}")
            
            # Detailed classification report
            report = classification_report(
                y_val, y_pred,
                target_names=['REJECT', 'Others'],
                output_dict=True
            )
            results[model_name]['classification_report'] = report
        
        # Select best model based on F1 score
        self.best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        logger.info(f"\nBest model: {self.best_model_name} (F1: {results[self.best_model_name]['f1_score']:.4f})")
        
        return results
    
    def predict(self, X: np.ndarray, return_proba: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the best model"""
        if self.best_model_name is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        X_scaled = self.scaler.transform(X)
        model = self.models[self.best_model_name]
        
        predictions = model.predict(X_scaled)
        
        if return_proba:
            probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (Others)
            return predictions, probabilities
        
        return predictions, None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances from the best model"""
        if self.best_model_name is None:
            return None
        
        model = self.models[self.best_model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(model, 'coef_'):
            # For linear models
            coefficients = np.abs(model.coef_[0])
            return dict(zip(self.feature_names, coefficients))
        
        return None
    
    def plot_performance(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = 'model1_performance.png'):
        """Plot model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get predictions from best model
        y_pred, y_proba = self.predict(X_test)
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['REJECT', 'Others'],
                   yticklabels=['REJECT', 'Others'])
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        ax2 = axes[0, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Importance
        ax3 = axes[1, 0]
        feature_importance = self.get_feature_importance()
        if feature_importance:
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_features)
            
            y_pos = np.arange(len(features))
            ax3.barh(y_pos, importances)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(features)
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importances')
        
        # 4. Precision-Recall Curve
        ax4 = axes[1, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        ax4.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curve')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {save_path}")
    
    def save_model(self, model_dir: str = 'model1_binary'):
        """Save the trained model"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save best model
        if self.best_model_name:
            joblib.dump(self.models[self.best_model_name], model_path / 'best_model.pkl')
            joblib.dump(self.best_model_name, model_path / 'best_model_name.pkl')
        
        # Save all models for comparison
        for model_name, model in self.models.items():
            joblib.dump(model, model_path / f'{model_name}.pkl')
        
        # Save scaler and metadata
        joblib.dump(self.scaler, model_path / 'scaler.pkl')
        joblib.dump(self.class_weights, model_path / 'class_weights.pkl')
        joblib.dump(self.feature_names, model_path / 'feature_names.pkl')
        
        logger.info(f"Model 1 saved to {model_path}")
    
    def load_model(self, model_dir: str = 'model1_binary'):
        """Load a trained model"""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise ValueError(f"Model directory {model_path} does not exist")
        
        # Load best model info
        self.best_model_name = joblib.load(model_path / 'best_model_name.pkl')
        
        # Load all models
        for model_file in model_path.glob('*.pkl'):
            if model_file.stem in ['rf_balanced', 'gb_classifier', 'logistic_balanced', 'svm_balanced']:
                self.models[model_file.stem] = joblib.load(model_file)
        
        # Load other components
        self.scaler = joblib.load(model_path / 'scaler.pkl')
        self.class_weights = joblib.load(model_path / 'class_weights.pkl')
        self.feature_names = joblib.load(model_path / 'feature_names.pkl')
        
        logger.info(f"Model 1 loaded from {model_path}")


if __name__ == "__main__":
    logger.info("Model 1: Binary Classifier (REJECT vs Others)")