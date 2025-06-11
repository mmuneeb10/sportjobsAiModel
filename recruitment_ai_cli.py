#!/usr/bin/env python3
"""
Command-line interface for the Recruitment AI System
Allows recruiters to evaluate new CVs against job descriptions
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from tabulate import tabulate

from cv_processor import CVProcessor
from advanced_recruitment_model import RecruitmentPipeline


class RecruitmentAICLI:
    """CLI for recruitment AI system"""
    
    def __init__(self):
        self.pipeline = RecruitmentPipeline()
        self.cv_processor = CVProcessor()
        
    def evaluate_single_cv(self, cv_path: str, job_desc_path: str) -> dict:
        """Evaluate a single CV against a job description"""
        print(f"\n{'=' * 60}")
        print(f"Evaluating CV: {Path(cv_path).name}")
        print(f"For position: {Path(job_desc_path).parent.name}")
        print(f"{'=' * 60}\n")
        
        try:
            # Process the application
            result = self.pipeline.process_new_application(cv_path, job_desc_path)
            
            # Display results
            self._display_results(result)
            
            return result
            
        except Exception as e:
            print(f"Error processing CV: {e}")
            return None
    
    def batch_evaluate(self, cv_folder: str, job_desc_path: str) -> list:
        """Evaluate multiple CVs from a folder"""
        cv_folder_path = Path(cv_folder)
        results = []
        
        print(f"\n{'=' * 60}")
        print(f"Batch Evaluation")
        print(f"CVs from: {cv_folder}")
        print(f"Job: {Path(job_desc_path).parent.name}")
        print(f"{'=' * 60}\n")
        
        # Find all CV files
        cv_files = []
        for ext in self.cv_processor.supported_formats:
            cv_files.extend(cv_folder_path.glob(f"*{ext}"))
        
        print(f"Found {len(cv_files)} CVs to evaluate\n")
        
        # Process each CV
        for cv_file in cv_files:
            try:
                result = self.pipeline.process_new_application(
                    str(cv_file), job_desc_path
                )
                results.append(result)
                
                # Brief summary for each
                print(f"✓ {cv_file.name:<30} → {result['prediction']:<12} "
                      f"(confidence: {result['confidence']:.2%})")
                
            except Exception as e:
                print(f"✗ {cv_file.name:<30} → ERROR: {e}")
        
        # Summary statistics
        self._display_batch_summary(results)
        
        return results
    
    def _display_results(self, result: dict):
        """Display detailed results for a single CV"""
        # Main prediction
        print(f"🎯 DECISION: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2%}\n")
        
        # Probability breakdown
        print("Stage Probabilities:")
        probs_data = [[stage, f"{prob:.2%}"] 
                      for stage, prob in result['probabilities'].items()]
        print(tabulate(probs_data, headers=['Stage', 'Probability'], 
                      tablefmt='simple'))
        
        # Candidate summary
        print("\nCandidate Summary:")
        summary = result['candidate_summary']
        summary_data = [
            ['Experience', f"{summary['experience_years']} years"],
            ['Skills', f"{summary['skills_count']} identified"],
            ['Certifications', summary['certifications']]
        ]
        print(tabulate(summary_data, tablefmt='simple'))
        
        # Decision explanation
        if 'explanation' in result and 'reasons' in result['explanation']:
            print("\nKey Decision Factors:")
            for reason in result['explanation']['reasons'][:5]:
                if 'importance' in reason:
                    print(f"  • {reason['feature']}: {reason['value']:.2f} "
                          f"(importance: {reason['importance']:.2%})")
                elif 'detail' in reason:
                    print(f"  • {reason['feature']}: {reason['detail']}")
    
    def _display_batch_summary(self, results: list):
        """Display summary of batch evaluation"""
        if not results:
            return
        
        print(f"\n{'=' * 60}")
        print("BATCH EVALUATION SUMMARY")
        print(f"{'=' * 60}\n")
        
        # Count by stage
        stage_counts = {'REJECT': 0, 'SHORTLIST': 0, 'INTERVIEW': 0, 'ACCEPT': 0}
        for result in results:
            stage_counts[result['prediction']] += 1
        
        # Display funnel
        total = len(results)
        funnel_data = []
        for stage, count in stage_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            bar = '█' * int(percentage / 2)
            funnel_data.append([stage, count, f"{percentage:.1f}%", bar])
        
        print(tabulate(funnel_data, 
                      headers=['Stage', 'Count', 'Percentage', 'Distribution'],
                      tablefmt='simple'))
        
        # Top candidates (highest accept probability)
        top_candidates = sorted(results, 
                               key=lambda x: x['probabilities']['ACCEPT'], 
                               reverse=True)[:5]
        
        print("\nTop 5 Candidates (by Accept probability):")
        top_data = []
        for i, candidate in enumerate(top_candidates, 1):
            cv_name = Path(candidate['cv_path']).name
            accept_prob = candidate['probabilities']['ACCEPT']
            prediction = candidate['prediction']
            top_data.append([i, cv_name[:30], prediction, f"{accept_prob:.2%}"])
        
        print(tabulate(top_data, 
                      headers=['Rank', 'CV', 'Prediction', 'Accept Prob'],
                      tablefmt='simple'))
    
    def export_results(self, results: list, output_file: str):
        """Export results to file"""
        output_path = Path(output_file)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif output_path.suffix == '.csv':
            # Convert to flat structure for CSV
            flat_results = []
            for result in results:
                flat = {
                    'cv_path': result['cv_path'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    **{f'prob_{k}': v for k, v in result['probabilities'].items()},
                    'experience_years': result['candidate_summary']['experience_years'],
                    'skills_count': result['candidate_summary']['skills_count']
                }
                flat_results.append(flat)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        print(f"\nResults exported to: {output_path}")
    
    def interactive_mode(self):
        """Interactive evaluation mode"""
        print("\n🤖 Recruitment AI - Interactive Mode")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("1. Evaluate single CV")
            print("2. Batch evaluate folder")
            print("3. Load different model")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                cv_path = input("Enter CV path: ").strip()
                job_path = input("Enter job description path: ").strip()
                
                if Path(cv_path).exists() and Path(job_path).exists():
                    self.evaluate_single_cv(cv_path, job_path)
                else:
                    print("Error: File not found")
                    
            elif choice == '2':
                cv_folder = input("Enter CV folder path: ").strip()
                job_path = input("Enter job description path: ").strip()
                
                if Path(cv_folder).exists() and Path(job_path).exists():
                    results = self.batch_evaluate(cv_folder, job_path)
                    
                    export = input("\nExport results? (y/n): ").strip().lower()
                    if export == 'y':
                        output_file = input("Output file (.json or .csv): ").strip()
                        self.export_results(results, output_file)
                else:
                    print("Error: Path not found")
                    
            elif choice == '3':
                model_path = input("Enter model path: ").strip()
                try:
                    self.pipeline.model.load_model(model_path)
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    
            elif choice == '4':
                print("\nGoodbye! 👋")
                break
            else:
                print("Invalid option")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Recruitment AI System - Evaluate CVs using AI trained on 20+ years of recruitment patterns'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a CV')
    eval_parser.add_argument('cv', help='Path to CV file')
    eval_parser.add_argument('job', help='Path to job description')
    eval_parser.add_argument('--model', help='Path to model directory', 
                           default='recruitment_model')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch evaluate CVs')
    batch_parser.add_argument('folder', help='Folder containing CVs')
    batch_parser.add_argument('job', help='Path to job description')
    batch_parser.add_argument('--output', help='Output file (.json or .csv)')
    batch_parser.add_argument('--model', help='Path to model directory',
                            default='recruitment_model')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', 
                                             help='Interactive mode')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--jobs-dir', help='Jobs directory', 
                            default='jobs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = RecruitmentAICLI()
    
    # Load model if specified
    if hasattr(args, 'model') and args.model:
        try:
            cli.pipeline.model.load_model(args.model)
        except Exception as e:
            print(f"Warning: Could not load model from {args.model}: {e}")
            print("Using default model or creating new one...")
    
    # Execute command
    if args.command == 'evaluate':
        cli.evaluate_single_cv(args.cv, args.job)
        
    elif args.command == 'batch':
        results = cli.batch_evaluate(args.folder, args.job)
        if args.output and results:
            cli.export_results(results, args.output)
            
    elif args.command == 'interactive':
        cli.interactive_mode()
        
    elif args.command == 'train':
        from train_recruitment_model import RecruitmentModelTrainer
        trainer = RecruitmentModelTrainer(args.jobs_dir)
        trainer.run_training_pipeline()


if __name__ == "__main__":
    main()