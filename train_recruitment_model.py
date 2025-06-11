#!/usr/bin/env python3
"""
Main training script for the recruitment AI model
Trains on historical recruitment data to learn patterns from 20+ years of experience
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from cv_processor import BatchCVProcessor
from advanced_recruitment_model import AdvancedRecruitmentModel


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RecruitmentModelTrainer:
    """Main training class for recruitment AI"""
    
    def __init__(self, jobs_directory: str):
        self.jobs_directory = Path(jobs_directory)
        self.batch_processor = BatchCVProcessor()
        self.model = AdvancedRecruitmentModel()
        self.training_data = None
        self.results = {}
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load all CV data from jobs directory"""
        logger.info(f"Loading data from {self.jobs_directory}")
        
        # Process all jobs
        df = self.batch_processor.process_all_jobs(str(self.jobs_directory))
        
        if df.empty:
            logger.warning("No CV data found. Creating sample data for demonstration...")
            df = self._create_sample_data()
        
        logger.info(f"Loaded {len(df)} CV samples")
        logger.info(f"Stage distribution:\n{df['stage'].value_counts()}")
        
        self.training_data = df
        return df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample training data for demonstration"""
        # This would be replaced with actual CV data
        sample_data = []
        
        stages = ['REJECT', 'SHORTLIST', 'INTERVIEW', 'ACCEPT']
        stage_distributions = [0.4, 0.3, 0.2, 0.1]  # Typical funnel
        
        for i in range(1000):  # Generate 1000 samples
            stage = np.random.choice(stages, p=stage_distributions)
            
            # Create realistic features based on stage
            if stage == 'REJECT':
                exp_years = np.random.randint(0, 3)
                skills = np.random.randint(1, 5)
                education = np.random.choice([0, 1], p=[0.7, 0.3])
            elif stage == 'SHORTLIST':
                exp_years = np.random.randint(2, 7)
                skills = np.random.randint(3, 8)
                education = np.random.choice([1, 2], p=[0.6, 0.4])
            elif stage == 'INTERVIEW':
                exp_years = np.random.randint(5, 12)
                skills = np.random.randint(5, 10)
                education = np.random.choice([2, 3], p=[0.7, 0.3])
            else:  # ACCEPT
                exp_years = np.random.randint(8, 20)
                skills = np.random.randint(8, 15)
                education = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            
            sample = {
                'stage': stage,
                'total_experience_years': exp_years,
                'skills': ['skill' + str(j) for j in range(skills)],
                'education': [{'degree': f'degree_{education}'}],
                'certifications': ['cert' + str(j) for j in range(np.random.randint(0, 3))],
                'languages': ['English'] + (['Spanish'] if np.random.random() > 0.7 else []),
                'raw_text': f"Sample CV text with {exp_years} years experience",
                'contact_info': {'location': 'Darwin' if np.random.random() > 0.3 else 'Other'},
                'job_description': "Sample job description for executive position"
            }
            
            sample_data.append(sample)
        
        return pd.DataFrame(sample_data)
    
    def prepare_features(self) -> tuple:
        """Prepare features and labels for training"""
        logger.info("Preparing features...")
        
        X, y = self.model.prepare_training_data(self.training_data)
        
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Label distribution: {np.bincount(y)}")
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train all models"""
        logger.info("Starting model training...")
        
        # Check if we have enough samples for stratification
        min_class_samples = np.min(np.bincount(y))
        logger.info(f"Minimum samples per class: {min_class_samples}")
        
        # Split data - use stratification only if we have enough samples
        if min_class_samples >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            logger.warning("Not enough samples for stratified split. Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train ensemble models
        logger.info("Training ensemble models...")
        self.model.train_ensemble(X_train, y_train)
        
        # Train neural network (optional - requires more data)
        # logger.info("Training neural network...")
        # self.model.train_neural_network(X_train, y_train)
        
        # Evaluate
        self.evaluate_models(X_test, y_test)
        
        # Store results
        self.results['training_samples'] = len(X_train)
        self.results['test_samples'] = len(X_test)
        
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance"""
        logger.info("Evaluating models...")
        
        # Get predictions
        y_pred, y_proba = self.model.predict_ensemble(X_test)
        
        # Determine actual stage names based on the classes present
        all_stage_names = ['REJECT', 'SHORTLIST', 'INTERVIEW', 'ACCEPT']
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        stage_names = [all_stage_names[i] for i in unique_classes if i < len(all_stage_names)]
        
        logger.info(f"Classes found in data: {unique_classes}")
        logger.info(f"Using stage names: {stage_names}")
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                     target_names=stage_names,
                                     labels=unique_classes,
                                     output_dict=True)
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=stage_names, labels=unique_classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=stage_names, yticklabels=stage_names)
        plt.title('Recruitment Decision Confusion Matrix')
        plt.ylabel('True Stage')
        plt.xlabel('Predicted Stage')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Store results
        self.results['accuracy'] = report['accuracy']
        self.results['classification_report'] = report
        self.results['confusion_matrix'] = cm.tolist()
        
    def analyze_feature_importance(self):
        """Analyze which features are most important"""
        logger.info("Analyzing feature importance...")
        
        # Get feature importances from Random Forest
        rf_model = self.model.models['rf']
        if hasattr(rf_model, 'feature_importances_'):
            feature_names = [
                'Text Similarity', 'Experience Years', 'Experience Ratio',
                'Meets Experience', 'Skill Match Ratio', 'Skill Coverage',
                'Total Skills', 'Matching Skills', 'Education Level',
                'Certifications', 'Industry Keywords', 'Communication Score',
                'Achievements', 'Languages', 'Location Match'
            ]
            
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            logger.info("\nTop 10 Most Important Features:")
            for i in range(min(10, len(indices))):
                if indices[i] < len(feature_names):
                    logger.info(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            top_features = [feature_names[i] for i in indices[:10] if i < len(feature_names)]
            top_importances = [importances[i] for i in indices[:10] if i < len(feature_names)]
            
            plt.barh(top_features, top_importances)
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importances for Recruitment Decisions')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            
            self.results['feature_importance'] = {
                feature_names[i]: float(importances[i]) 
                for i in range(len(feature_names)) if i < len(importances)
            }
    
    def save_results(self):
        """Save training results and model"""
        logger.info("Saving results...")
        
        # Save model
        self.model.save_model('recruitment_model')
        
        # Save training results
        with open('training_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save training data summary
        if self.training_data is not None:
            self.training_data[['stage', 'total_experience_years']].to_csv(
                'training_data_summary.csv', index=False
            )
        
        logger.info("Results saved successfully!")
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        logger.info("=" * 50)
        logger.info("Starting Recruitment AI Training Pipeline")
        logger.info("=" * 50)
        
        # Load data
        self.load_and_preprocess_data()
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Train models
        self.train_models(X, y)
        
        # Analyze features
        self.analyze_feature_importance()
        
        # Save everything
        self.save_results()
        
        logger.info("=" * 50)
        logger.info("Training completed successfully!")
        logger.info(f"Overall accuracy: {self.results['accuracy']:.2%}")
        logger.info("=" * 50)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Recruitment AI Model')
    parser.add_argument('--jobs-dir', type=str, default='jobs',
                       help='Directory containing job folders')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RecruitmentModelTrainer(args.jobs_dir)
    
    if args.evaluate_only:
        # Load existing model and evaluate
        logger.info("Loading existing model for evaluation...")
        trainer.model.load_model('recruitment_model')
        # Evaluate on test data
    else:
        # Run full training pipeline
        trainer.run_training_pipeline()


if __name__ == "__main__":
    main()