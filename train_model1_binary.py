#!/usr/bin/env python3
"""
Training Script for Model 1: Binary Classifier
Trains a binary classifier to distinguish between REJECT and Others (SHORTLIST/INTERVIEW/ACCEPT)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

# Import necessary modules
from cv_processor import BatchCVProcessor
from advanced_recruitment_model import AdvancedRecruitmentModel
from model1_binary_classifier import Model1BinaryClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model1_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Model1Trainer:
    """Trainer for Model 1 (Binary Classifier)"""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.batch_processor = BatchCVProcessor()
        self.feature_extractor = AdvancedRecruitmentModel()  # For feature extraction
        self.model1 = Model1BinaryClassifier()
        self.training_data = None
        self.results = {}
        
    def load_training_data(self) -> pd.DataFrame:
        """Load all CV data from the specified directory"""
        logger.info(f"Loading training data from {self.data_directory}")
        
        # Check if directory exists
        if not self.data_directory.exists():
            logger.error(f"Directory {self.data_directory} does not exist!")
            raise ValueError(f"Directory {self.data_directory} does not exist!")
        
        # Get all job folders
        job_folders = [d for d in self.data_directory.iterdir() 
                      if d.is_dir() and (d / 'job_description.txt').exists()]
        
        logger.info(f"Found {len(job_folders)} job folders")
        
        # Process all jobs using batch processor
        df = self.batch_processor.process_all_jobs(str(self.data_directory))
        
        if df.empty:
            logger.error("No CV data found in the directory!")
            raise ValueError("No CV data found. Please check your data directory.")
        
        # Log data statistics
        logger.info(f"Loaded {len(df)} CV samples")
        stage_counts = df['stage'].value_counts()
        logger.info("Stage distribution:")
        for stage, count in stage_counts.items():
            logger.info(f"  {stage}: {count} ({count/len(df)*100:.1f}%)")
        
        # Calculate binary distribution
        reject_count = stage_counts.get('REJECT', 0)
        others_count = len(df) - reject_count
        
        logger.info("\nBinary distribution:")
        logger.info(f"  REJECT: {reject_count} ({reject_count/len(df)*100:.1f}%)")
        logger.info(f"  Others: {others_count} ({others_count/len(df)*100:.1f}%)")
        
        self.training_data = df
        return df
    
    def prepare_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features and prepare binary labels"""
        logger.info("Extracting features and preparing labels...")
        
        # Use the advanced model to extract features
        X, y_multiclass = self.feature_extractor.prepare_training_data(self.training_data)
        
        # Get stage names for conversion
        stage_names = self.training_data['stage'].values
        
        # Convert to binary labels
        y_binary = self.model1.convert_labels_to_binary(stage_names)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Binary labels: {np.bincount(y_binary)}")
        
        return X, y_binary, stage_names
    
    def train_model(self):
        """Main training function"""
        logger.info("="*70)
        logger.info("TRAINING MODEL 1: BINARY CLASSIFIER")
        logger.info("="*70)
        
        # Load data
        self.load_training_data()
        
        # Prepare features and labels
        X, y_binary, stage_names = self.prepare_features_and_labels()
        
        # Split data into train/validation/test sets
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Second split: 75% train, 25% validation (of the 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train the model
        training_results = self.model1.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        self.evaluate_on_test_set(X_test, y_test)
        
        # Cross-validation for robustness
        self.perform_cross_validation(X_temp, y_temp)
        
        # Plot performance
        self.model1.plot_performance(X_test, y_test, 'model1_performance.png')
        
        # Save results
        self.save_training_results(training_results, X_test, y_test)
        
        # Save the model
        self.model1.save_model('model1_binary')
        
        logger.info("\n" + "="*70)
        logger.info("MODEL 1 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model on test set"""
        y_pred, y_proba = self.model1.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Detailed classification report
        logger.info("\nClassification Report:")
        report = classification_report(
            y_test, y_pred,
            target_names=['REJECT', 'Others'],
            digits=4
        )
        logger.info(report)
        
        # Store results
        self.results['test_performance'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'balanced_accuracy': balanced_acc
        }
    
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray):
        """Perform k-fold cross-validation"""
        logger.info("\nPerforming 5-fold cross-validation...")
        
        model = self.model1.models[self.model1.best_model_name]
        cv_scores = cross_val_score(
            model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1', n_jobs=-1
        )
        
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.results['cross_validation'] = {
            'scores': cv_scores.tolist(),
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
    
    def save_training_results(self, training_results: dict, X_test: np.ndarray, y_test: np.ndarray):
        """Save all training results"""
        # Get feature importance
        feature_importance = self.model1.get_feature_importance()
        
        # Clean training results to remove numpy arrays
        cleaned_results = {}
        for model_name, metrics in training_results.items():
            cleaned_results[model_name] = {
                'f1_score': float(metrics['f1_score']),
                'roc_auc': float(metrics['roc_auc']),
                'balanced_accuracy': float(metrics['balanced_accuracy']),
                'classification_report': metrics['classification_report']
            }
        
        # Compile results
        self.results.update({
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Binary Classifier (REJECT vs Others)',
            'best_model': self.model1.best_model_name,
            'total_samples': int(len(self.training_data)),
            'stage_distribution': self.training_data['stage'].value_counts().to_dict(),
            'model_comparison': cleaned_results,
            'feature_importance': feature_importance,
            'class_weights': {str(k): float(v) for k, v in self.model1.class_weights.items()} if self.model1.class_weights else None
        })
        
        # Save to JSON
        with open('model1_training_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save data summary
        summary_df = pd.DataFrame({
            'stage': self.training_data['stage'].value_counts().index,
            'count': self.training_data['stage'].value_counts().values,
            'percentage': (self.training_data['stage'].value_counts().values / len(self.training_data) * 100).round(2)
        })
        summary_df.to_csv('model1_data_summary.csv', index=False)
        
        logger.info("Training results saved!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train Model 1: Binary Classifier (REJECT vs Others)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train using 'data' directory
  python train_model1_binary.py --data-dir data/
  
  # Train using 'jobs' directory
  python train_model1_binary.py --data-dir jobs/
  
  # Train using custom directory
  python train_model1_binary.py --data-dir /path/to/your/data/
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing the training data (job folders with CVs)'
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        # Try 'jobs' directory as fallback
        jobs_path = Path('jobs')
        if jobs_path.exists():
            logger.info(f"Data directory '{args.data_dir}' not found. Using 'jobs' directory instead.")
            args.data_dir = 'jobs'
        else:
            logger.error(f"Data directory '{args.data_dir}' not found!")
            logger.error("Please specify a valid directory containing job folders with CVs.")
            sys.exit(1)
    
    # Create trainer and run training
    trainer = Model1Trainer(args.data_dir)
    
    try:
        trainer.train_model()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()