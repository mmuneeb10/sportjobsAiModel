#!/usr/bin/env python3
"""
Model 2: Three-Class Classifier
This model classifies candidates into:
- Class 0: SHORTLIST
- Class 1: INTERVIEW
- Class 2: ACCEPT

Note: This model is trained ONLY on candidates that are NOT rejected.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, balanced_accuracy_score, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model2ThreeClassClassifier:
    """Three-class classifier for SHORTLIST/INTERVIEW/ACCEPT"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model_name = None
        self.class_weights = None
        
        # 3-class mapping (excluding REJECT)
        self.label_mapping = {
            'SHORTLIST': 0,
            'INTERVIEW': 1,
            'ACCEPT': 2
        }
        
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Feature names for interpretability
        self.feature_names = [
            'text_similarity', 'experience_years', 'experience_ratio',
            'meets_experience', 'skill_match_ratio', 'skill_coverage',
            'total_skills', 'matching_skills', 'education_level',
            'certifications', 'industry_keywords', 'communication_score',
            'achievements', 'languages', 'location_match'
        ]
        
        # Initialize multiple models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different models optimized for 3-class classification"""
        
        # Random Forest with balanced weights
        self.models['rf_balanced'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees for better generalization
        self.models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['gb_classifier'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            random_state=42
        )
        
        # Logistic Regression (One-vs-Rest)
        self.models['logistic_ovr'] = LogisticRegression(
            multi_class='ovr',
            class_weight='balanced',
            max_iter=2000,
            C=0.5,
            random_state=42,
            solver='liblinear'
        )
        
        # Neural Network
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
    
    def filter_non_reject_data(self, stages: List[str]) -> np.ndarray:
        """Filter out REJECT candidates and return indices"""
        non_reject_indices = [i for i, stage in enumerate(stages) if stage != 'REJECT']
        return np.array(non_reject_indices)
    
    def convert_labels_to_three_class(self, stages: List[str]) -> np.ndarray:
        """Convert stage labels to 3-class format (excluding REJECT)"""
        labels = []
        for stage in stages:
            if stage in self.label_mapping:
                labels.append(self.label_mapping[stage])
            else:
                # This shouldn't happen if data is filtered properly
                logger.warning(f"Unexpected stage: {stage}")
        
        return np.array(labels)
    
    def prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features"""
        return self.scaler.fit_transform(X)
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for the 3 classes"""
        unique_classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
        
        self.class_weights = dict(zip(unique_classes, class_weights))
        
        # Log class distribution and weights
        logger.info("Class distribution and weights:")
        for class_idx in unique_classes:
            count = np.sum(y == class_idx)
            weight = self.class_weights[class_idx]
            class_name = self.reverse_mapping[class_idx]
            logger.info(f"  {class_name}: {count} samples, weight: {weight:.3f}")
        
        return self.class_weights
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train all models and select the best one"""
        logger.info("Training Model 2 (3-Class Classifier)...")
        
        # Compute class weights
        self.compute_class_weights(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train each model
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\nTraining {model_name}...")
            
            # Train model
            if model_name == 'gb_classifier' and self.class_weights:
                # Gradient Boosting doesn't support class_weight, use sample_weight
                sample_weights = np.array([self.class_weights[y] for y in y_train])
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            elif model_name == 'mlp':
                # Neural network benefits from balanced batches
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average='macro')
            f1_weighted = f1_score(y_val, y_pred, average='weighted')
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_val, y_pred, average=None, labels=[0, 1, 2]
            )
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'balanced_accuracy': balanced_acc,
                'predictions': y_pred,
                'per_class_metrics': {
                    self.reverse_mapping[i]: {
                        'precision': precision[i],
                        'recall': recall[i],
                        'f1': f1[i],
                        'support': support[i]
                    } for i in range(len(precision))
                }
            }
            
            # Log performance
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1 (macro): {f1_macro:.4f}, Balanced Acc: {balanced_acc:.4f}")
            
            # Detailed classification report
            report = classification_report(
                y_val, y_pred,
                target_names=['SHORTLIST', 'INTERVIEW', 'ACCEPT'],
                output_dict=True
            )
            results[model_name]['classification_report'] = report
        
        # Select best model based on macro F1 score (good for multi-class)
        self.best_model_name = max(results, key=lambda x: results[x]['f1_macro'])
        logger.info(f"\nBest model: {self.best_model_name} (F1 macro: {results[self.best_model_name]['f1_macro']:.4f})")
        
        return results
    
    def predict(self, X: np.ndarray, return_proba: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the best model"""
        if self.best_model_name is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        X_scaled = self.scaler.transform(X)
        model = self.models[self.best_model_name]
        
        predictions = model.predict(X_scaled)
        
        if return_proba and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
            return predictions, probabilities
        
        return predictions, None
    
    def predict_stage_names(self, X: np.ndarray) -> List[str]:
        """Predict and return stage names instead of numeric labels"""
        predictions, _ = self.predict(X, return_proba=False)
        return [self.reverse_mapping[pred] for pred in predictions]
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances from the best model"""
        if self.best_model_name is None:
            return None
        
        model = self.models[self.best_model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(model, 'coef_'):
            # For linear models, average absolute coefficients across classes
            coefficients = np.mean(np.abs(model.coef_), axis=0)
            return dict(zip(self.feature_names, coefficients))
        
        return None
    
    def plot_performance(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = 'model2_performance.png'):
        """Plot model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get predictions from best model
        y_pred, y_proba = self.predict(X_test)
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['SHORTLIST', 'INTERVIEW', 'ACCEPT'],
                   yticklabels=['SHORTLIST', 'INTERVIEW', 'ACCEPT'])
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Per-class Performance
        ax2 = axes[0, 1]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=[0, 1, 2]
        )
        
        x = np.arange(3)
        width = 0.25
        
        ax2.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax2.bar(x, recall, width, label='Recall', alpha=0.8)
        ax2.bar(x + width, f1, width, label='F1', alpha=0.8)
        
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Score')
        ax2.set_title('Per-class Performance Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['SHORTLIST', 'INTERVIEW', 'ACCEPT'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Importance
        ax3 = axes[1, 0]
        feature_importance = self.get_feature_importance()
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_features)
            
            y_pos = np.arange(len(features))
            ax3.barh(y_pos, importances)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(features)
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importances')
        
        # 4. Class Distribution in Predictions
        ax4 = axes[1, 1]
        unique, counts = np.unique(y_pred, return_counts=True)
        true_unique, true_counts = np.unique(y_test, return_counts=True)
        
        x = np.arange(3)
        width = 0.35
        
        ax4.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        ax4.bar(x + width/2, counts, width, label='Predicted', alpha=0.8)
        
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Count')
        ax4.set_title('Class Distribution: True vs Predicted')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['SHORTLIST', 'INTERVIEW', 'ACCEPT'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {save_path}")
    
    def save_model(self, model_dir: str = 'model2_three_class'):
        """Save the trained model"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save best model
        if self.best_model_name:
            joblib.dump(self.models[self.best_model_name], model_path / 'best_model.pkl')
            joblib.dump(self.best_model_name, model_path / 'best_model_name.pkl')
        
        # Save all models
        for model_name, model in self.models.items():
            joblib.dump(model, model_path / f'{model_name}.pkl')
        
        # Save scaler and metadata
        joblib.dump(self.scaler, model_path / 'scaler.pkl')
        joblib.dump(self.class_weights, model_path / 'class_weights.pkl')
        joblib.dump(self.feature_names, model_path / 'feature_names.pkl')
        joblib.dump(self.label_mapping, model_path / 'label_mapping.pkl')
        
        logger.info(f"Model 2 saved to {model_path}")
    
    def load_model(self, model_dir: str = 'model2_three_class'):
        """Load a trained model"""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise ValueError(f"Model directory {model_path} does not exist")
        
        # Load best model info
        self.best_model_name = joblib.load(model_path / 'best_model_name.pkl')
        
        # Load models
        for model_file in model_path.glob('*.pkl'):
            if model_file.stem in ['rf_balanced', 'extra_trees', 'gb_classifier', 'logistic_ovr', 'mlp']:
                self.models[model_file.stem] = joblib.load(model_file)
        
        # Load other components
        self.scaler = joblib.load(model_path / 'scaler.pkl')
        self.class_weights = joblib.load(model_path / 'class_weights.pkl')
        self.feature_names = joblib.load(model_path / 'feature_names.pkl')
        self.label_mapping = joblib.load(model_path / 'label_mapping.pkl')
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        logger.info(f"Model 2 loaded from {model_path}")


if __name__ == "__main__":
    logger.info("Model 2: Three-Class Classifier (SHORTLIST/INTERVIEW/ACCEPT)")