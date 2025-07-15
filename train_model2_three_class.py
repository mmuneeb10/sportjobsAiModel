#!/usr/bin/env python3
"""
Training Script for Model 2: Three-Class Classifier
Trains a 3-class classifier on ONLY non-rejected candidates (SHORTLIST/INTERVIEW/ACCEPT)
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
from model2_three_class_classifier import Model2ThreeClassClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model2_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Model2Trainer:
    """Trainer for Model 2 (3-Class Classifier)"""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.batch_processor = BatchCVProcessor()
        self.feature_extractor = AdvancedRecruitmentModel()  # For feature extraction
        self.model2 = Model2ThreeClassClassifier()
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
        
        # Log initial data statistics
        logger.info(f"Loaded {len(df)} total CV samples")
        stage_counts = df['stage'].value_counts()
        logger.info("Initial stage distribution:")
        for stage, count in stage_counts.items():
            logger.info(f"  {stage}: {count} ({count/len(df)*100:.1f}%)")
        
        # Filter out REJECT candidates for Model 2
        df_filtered = df[df['stage'] != 'REJECT'].copy()
        
        if df_filtered.empty:
            logger.error("No non-rejected candidates found!")
            raise ValueError("No suitable training data for Model 2")
        
        # Log filtered data statistics
        logger.info(f"\nAfter filtering REJECT candidates:")
        logger.info(f"Remaining samples: {len(df_filtered)}")
        
        filtered_counts = df_filtered['stage'].value_counts()
        logger.info("Filtered stage distribution:")
        for stage, count in filtered_counts.items():
            logger.info(f"  {stage}: {count} ({count/len(df_filtered)*100:.1f}%)")
        
        self.training_data = df_filtered
        return df_filtered
    
    def prepare_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features and prepare 3-class labels"""
        logger.info("Extracting features and preparing labels for Model 2...")
        
        # Get all data including REJECT for feature extraction
        all_data = self.batch_processor.process_all_jobs(str(self.data_directory))
        
        # Use the advanced model to extract features for all data
        X_all, y_all = self.feature_extractor.prepare_training_data(all_data)
        
        # Get indices of non-REJECT candidates
        non_reject_indices = all_data['stage'] != 'REJECT'
        
        # Filter features and stages
        X = X_all[non_reject_indices]
        stage_names = all_data[non_reject_indices]['stage'].values
        
        # Convert to 3-class labels
        y_three_class = self.model2.convert_labels_to_three_class(stage_names)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Three-class label distribution: {np.bincount(y_three_class)}")
        
        return X, y_three_class, stage_names
    
    def train_model(self):
        """Main training function"""
        logger.info("="*70)
        logger.info("TRAINING MODEL 2: THREE-CLASS CLASSIFIER")
        logger.info("(SHORTLIST/INTERVIEW/ACCEPT - No REJECT data)")
        logger.info("="*70)
        
        # Load data (already filtered)
        self.load_training_data()
        
        # Prepare features and labels
        X, y_three_class, stage_names = self.prepare_features_and_labels()
        
        # Check if we have all 3 classes
        unique_classes = np.unique(y_three_class)
        if len(unique_classes) < 3:
            logger.warning(f"Only {len(unique_classes)} classes found in data!")
            missing_classes = set([0, 1, 2]) - set(unique_classes)
            for missing in missing_classes:
                logger.warning(f"  Missing class: {self.model2.reverse_mapping.get(missing, missing)}")
        
        # Split data into train/validation/test sets
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_three_class, test_size=0.2, random_state=42, 
            stratify=y_three_class if len(unique_classes) > 1 else None
        )
        
        # Second split: 75% train, 25% validation (of the 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Log class distribution in each set
        logger.info("\nClass distribution in splits:")
        for name, y_set in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            unique, counts = np.unique(y_set, return_counts=True)
            dist = {self.model2.reverse_mapping.get(u, u): c for u, c in zip(unique, counts)}
            logger.info(f"  {name}: {dist}")
        
        # Train the model
        training_results = self.model2.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        self.evaluate_on_test_set(X_test, y_test)
        
        # Cross-validation for robustness
        if len(unique_classes) > 1:
            self.perform_cross_validation(X_temp, y_temp)
        
        # Plot performance
        self.model2.plot_performance(X_test, y_test, 'model2_performance.png')
        
        # Save results
        self.save_training_results(training_results, X_test, y_test)
        
        # Save the model
        self.model2.save_model('model2_three_class')
        
        logger.info("\n" + "="*70)
        logger.info("MODEL 2 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model on test set"""
        y_pred, y_proba = self.model2.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score (macro): {f1_macro:.4f}")
        logger.info(f"  F1 Score (weighted): {f1_weighted:.4f}")
        logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Detailed classification report
        logger.info("\nClassification Report:")
        
        # Get actual class names for the test set
        test_classes = np.unique(y_test)
        target_names = [self.model2.reverse_mapping.get(c, str(c)) for c in test_classes]
        
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            labels=test_classes,
            digits=4
        )
        logger.info(report)
        
        # Store results
        self.results['test_performance'] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'balanced_accuracy': balanced_acc
        }
    
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray):
        """Perform k-fold cross-validation"""
        logger.info("\nPerforming 5-fold cross-validation...")
        
        model = self.model2.models[self.model2.best_model_name]
        
        # Use fewer folds if we have limited data
        n_splits = min(5, len(np.unique(y)))
        
        try:
            cv_scores = cross_val_score(
                model, X, y, 
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                scoring='f1_macro', 
                n_jobs=-1
            )
            
            logger.info(f"Cross-validation F1 (macro) scores: {cv_scores}")
            logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            self.results['cross_validation'] = {
                'scores': cv_scores.tolist(),
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            self.results['cross_validation'] = {'error': str(e)}
    
    def save_training_results(self, training_results: dict, X_test: np.ndarray, y_test: np.ndarray):
        """Save all training results"""
        # Get feature importance
        feature_importance = self.model2.get_feature_importance()
        
        # Clean training results to remove numpy arrays
        cleaned_results = {}
        for model_name, metrics in training_results.items():
            # Clean per_class_metrics to handle numpy types
            per_class_cleaned = {}
            if 'per_class_metrics' in metrics:
                for class_name, class_metrics in metrics['per_class_metrics'].items():
                    per_class_cleaned[class_name] = {
                        'precision': float(class_metrics['precision']),
                        'recall': float(class_metrics['recall']),
                        'f1': float(class_metrics['f1']),
                        'support': int(class_metrics['support'])
                    }
            
            cleaned_results[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted']),
                'balanced_accuracy': float(metrics['balanced_accuracy']),
                'classification_report': metrics['classification_report'],
                'per_class_metrics': per_class_cleaned
            }
        
        # Compile results
        self.results.update({
            'timestamp': datetime.now().isoformat(),
            'model_type': '3-Class Classifier (SHORTLIST/INTERVIEW/ACCEPT)',
            'note': 'Trained only on non-rejected candidates',
            'best_model': self.model2.best_model_name,
            'total_samples': int(len(self.training_data)),
            'stage_distribution': self.training_data['stage'].value_counts().to_dict(),
            'model_comparison': cleaned_results,
            'feature_importance': feature_importance,
            'class_weights': {str(k): float(v) for k, v in self.model2.class_weights.items()} if self.model2.class_weights else None
        })
        
        # Save to JSON
        with open('model2_training_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save data summary
        summary_df = pd.DataFrame({
            'stage': self.training_data['stage'].value_counts().index,
            'count': self.training_data['stage'].value_counts().values,
            'percentage': (self.training_data['stage'].value_counts().values / len(self.training_data) * 100).round(2)
        })
        summary_df.to_csv('model2_data_summary.csv', index=False)
        
        logger.info("Training results saved!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train Model 2: Three-Class Classifier (SHORTLIST/INTERVIEW/ACCEPT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script trains Model 2 which classifies candidates into 3 stages:
- SHORTLIST
- INTERVIEW  
- ACCEPT

IMPORTANT: This model is trained ONLY on candidates that were NOT rejected.
           All REJECT candidates are filtered out before training.

Examples:
  # Train using 'data' directory
  python train_model2_three_class.py --data-dir data/
  
  # Train using 'jobs' directory
  python train_model2_three_class.py --data-dir jobs/
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
    trainer = Model2Trainer(args.data_dir)
    
    try:
        trainer.train_model()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()