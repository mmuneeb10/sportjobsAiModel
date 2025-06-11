import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class JobPosting:
    job_id: str
    title: str
    description: str
    requirements: List[str]
    responsibilities: List[str]
    skills: List[str]
    experience_years: Optional[int]
    location: str


@dataclass
class Candidate:
    cv_path: str
    content: str
    experience_years: int
    skills: List[str]
    education: List[str]
    current_stage: str  # REJECT, SHORTLIST, INTERVIEW, ACCEPT


class RecruitmentAISystem:
    def __init__(self, jobs_directory: str):
        self.jobs_directory = Path(jobs_directory)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.skill_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.stage_mapping = {
            'REJECT': 0,
            'SHORTLIST': 1,
            'INTERVIEW': 2,
            'ACCEPT': 3
        }
        
    def parse_job_description(self, job_desc_path: str) -> JobPosting:
        """Parse job description file and extract key information"""
        with open(job_desc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract job title
        lines = content.strip().split('\n')
        title = lines[0] if lines else ""
        
        # Extract key sections using pattern matching
        description = content
        
        # Extract requirements (common patterns)
        req_patterns = [
            r'requirements?:?(.*?)(?=responsibilities|experience|skills|$)',
            r'you will have:?(.*?)(?=responsibilities|experience|skills|$)',
            r'essential criteria:?(.*?)(?=desirable|responsibilities|$)'
        ]
        requirements = []
        for pattern in req_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                requirements.extend([r.strip() for r in match.group(1).split('•') if r.strip()])
        
        # Extract skills
        skill_keywords = ['commercial acumen', 'leadership', 'stakeholder management', 
                         'strategic planning', 'financial management', 'negotiation',
                         'people management', 'business development']
        skills = [skill for skill in skill_keywords if skill.lower() in content.lower()]
        
        # Extract experience requirements
        exp_match = re.search(r'(\d+)\+?\s*years?', content, re.IGNORECASE)
        experience_years = int(exp_match.group(1)) if exp_match else None
        
        # Extract location
        location_match = re.search(r'(darwin|northern territory|nt)', content, re.IGNORECASE)
        location = location_match.group(0) if location_match else "Not specified"
        
        return JobPosting(
            job_id=os.path.basename(os.path.dirname(job_desc_path)),
            title=title,
            description=description,
            requirements=requirements,
            responsibilities=[],  # Could be extracted similarly
            skills=skills,
            experience_years=experience_years,
            location=location
        )
    
    def parse_cv(self, cv_path: str, stage: str) -> Optional[Candidate]:
        """Parse CV file and extract candidate information"""
        try:
            # Handle different file formats (for now assuming text-based)
            with open(cv_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract years of experience
            exp_patterns = [
                r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
                r'experience:\s*(\d+)\+?\s*years?',
            ]
            experience_years = 0
            for pattern in exp_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    experience_years = int(match.group(1))
                    break
            
            # Extract skills (simplified - in production, use NLP)
            common_skills = ['leadership', 'management', 'strategic planning', 
                           'stakeholder engagement', 'financial management',
                           'business development', 'negotiation', 'communication']
            found_skills = [skill for skill in common_skills if skill.lower() in content.lower()]
            
            # Extract education
            education = []
            edu_patterns = ['bachelor', 'master', 'mba', 'phd', 'diploma']
            for pattern in edu_patterns:
                if pattern.lower() in content.lower():
                    education.append(pattern)
            
            return Candidate(
                cv_path=cv_path,
                content=content,
                experience_years=experience_years,
                skills=found_skills,
                education=education,
                current_stage=stage
            )
        except Exception as e:
            print(f"Error parsing CV {cv_path}: {e}")
            return None
    
    def extract_features(self, job: JobPosting, candidate: Candidate) -> np.ndarray:
        """Extract features for ML model"""
        features = []
        
        # 1. Text similarity between job description and CV
        job_text = job.description
        cv_text = candidate.content
        
        if hasattr(self, 'fitted_vectorizer'):
            job_vec = self.vectorizer.transform([job_text])
            cv_vec = self.vectorizer.transform([cv_text])
        else:
            # Fit on first use
            self.vectorizer.fit([job_text, cv_text])
            self.fitted_vectorizer = True
            job_vec = self.vectorizer.transform([job_text])
            cv_vec = self.vectorizer.transform([cv_text])
        
        text_similarity = cosine_similarity(job_vec, cv_vec)[0][0]
        features.append(text_similarity)
        
        # 2. Experience match
        if job.experience_years:
            exp_diff = candidate.experience_years - job.experience_years
            exp_match = 1.0 if exp_diff >= 0 else max(0, 1 + exp_diff/job.experience_years)
        else:
            exp_match = min(1.0, candidate.experience_years / 10)  # Normalize to 10 years
        features.append(exp_match)
        
        # 3. Skills match
        if job.skills:
            skill_match = len(set(candidate.skills) & set(job.skills)) / len(job.skills)
        else:
            skill_match = len(candidate.skills) / 10  # Normalize
        features.append(skill_match)
        
        # 4. Education level
        edu_score = len(candidate.education) / 4  # Normalize to 4 levels
        features.append(edu_score)
        
        # 5. Keyword matches for specific domains
        keywords = ['CEO', 'executive', 'director', 'leadership', 'strategic']
        keyword_count = sum(1 for kw in keywords if kw.lower() in candidate.content.lower())
        keyword_score = keyword_count / len(keywords)
        features.append(keyword_score)
        
        return np.array(features)
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all historical data from jobs folders"""
        X = []
        y = []
        
        for job_folder in self.jobs_directory.iterdir():
            if not job_folder.is_dir():
                continue
                
            job_desc_path = job_folder / "job_description.txt"
            if not job_desc_path.exists():
                continue
                
            job = self.parse_job_description(str(job_desc_path))
            
            # Process CVs from each stage
            for stage in ['REJECT', 'SHORTLIST', 'INTERVIEW', 'ACCEPT']:
                stage_folder = job_folder / stage
                if not stage_folder.exists():
                    continue
                    
                for cv_file in stage_folder.iterdir():
                    if cv_file.is_file():
                        candidate = self.parse_cv(str(cv_file), stage)
                        if candidate:
                            features = self.extract_features(job, candidate)
                            X.append(features)
                            y.append(self.stage_mapping[stage])
        
        return np.array(X) if X else np.array([]).reshape(0, 5), np.array(y)
    
    def train(self):
        """Train the recruitment model"""
        print("Loading training data...")
        X, y = self.load_training_data()
        
        if len(X) == 0:
            print("No training data found. Please add CVs to the folders.")
            return
        
        print(f"Found {len(X)} CV samples across all jobs")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.2f}")
        print(f"Test accuracy: {test_score:.2f}")
        
        # Save model
        self.save_model()
    
    def predict_cv_stage(self, job_desc_path: str, cv_path: str) -> Tuple[str, float]:
        """Predict which stage a CV should be placed in"""
        job = self.parse_job_description(job_desc_path)
        candidate = self.parse_cv(cv_path, "UNKNOWN")
        
        if not candidate:
            return "REJECT", 0.0
        
        features = self.extract_features(job, candidate)
        features_scaled = self.scaler.transform([features])
        
        # Get prediction and probability
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Reverse mapping
        stage_names = {v: k for k, v in self.stage_mapping.items()}
        predicted_stage = stage_names[prediction]
        confidence = probabilities[prediction]
        
        return predicted_stage, confidence
    
    def save_model(self):
        """Save trained model and preprocessors"""
        model_dir = Path("recruitment_model")
        model_dir.mkdir(exist_ok=True)
        
        joblib.dump(self.classifier, model_dir / "classifier.pkl")
        joblib.dump(self.scaler, model_dir / "scaler.pkl")
        joblib.dump(self.vectorizer, model_dir / "vectorizer.pkl")
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self):
        """Load pre-trained model"""
        model_dir = Path("recruitment_model")
        
        if not model_dir.exists():
            raise ValueError("No saved model found. Please train first.")
        
        self.classifier = joblib.load(model_dir / "classifier.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.vectorizer = joblib.load(model_dir / "vectorizer.pkl")
        self.fitted_vectorizer = True
        
        print("Model loaded successfully")
    
    def analyze_recruitment_patterns(self):
        """Analyze patterns in recruitment decisions"""
        X, y = self.load_training_data()
        
        if len(X) == 0:
            print("No data to analyze")
            return
        
        # Feature importance
        if hasattr(self.classifier, 'feature_importances_'):
            feature_names = ['Text Similarity', 'Experience Match', 
                           'Skills Match', 'Education Score', 'Keyword Score']
            importances = self.classifier.feature_importances_
            
            print("\nFeature Importances:")
            for name, imp in zip(feature_names, importances):
                print(f"{name}: {imp:.3f}")
        
        # Stage distribution
        stage_names = {v: k for k, v in self.stage_mapping.items()}
        print("\nStage Distribution:")
        for stage_id in np.unique(y):
            count = np.sum(y == stage_id)
            print(f"{stage_names[stage_id]}: {count} ({count/len(y)*100:.1f}%)")


# Example usage
if __name__ == "__main__":
    # Initialize system
    ai_system = RecruitmentAISystem("jobs")
    
    # Train on historical data
    print("Training AI Recruitment System...")
    ai_system.train()
    
    # Analyze patterns
    print("\nAnalyzing recruitment patterns...")
    ai_system.analyze_recruitment_patterns()
    
    # Example: Predict for a new CV
    # prediction, confidence = ai_system.predict_cv_stage(
    #     "jobs/252_Chief Executive Officer 241212-01/job_description.txt",
    #     "new_cv.pdf"
    # )
    # print(f"\nPredicted stage: {prediction} (confidence: {confidence:.2f})")