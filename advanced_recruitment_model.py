import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import joblib
from pathlib import Path
import json
import re
from cv_processor import CVProcessor


class BERTFeatureExtractor:
    """Extract semantic features using BERT embeddings"""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def extract_embeddings(self, text: str, max_length: int = 512) -> np.ndarray:
        """Extract BERT embeddings for text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            padding=True,
            truncation=True
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings.flatten()


class RecruitmentNeuralNetwork(nn.Module):
    """Custom neural network for recruitment decisions"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int = 4):
        super(RecruitmentNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class AdvancedRecruitmentModel:
    """Advanced AI model combining multiple techniques"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=200, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        self.bert_extractor = None  # Initialize when needed
        self.neural_net = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_advanced_features(self, cv_data: Dict, job_data: Dict) -> np.ndarray:
        """Extract advanced features from CV and job description"""
        features = []
        
        # 1. Basic text similarity features
        cv_text = cv_data.get('raw_text', '')
        job_text = job_data.get('description', '')
        
        # TF-IDF similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform([cv_text, job_text])
            similarity = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0, 0]
            features.append(similarity)
        except:
            features.append(0)
        
        # 2. Experience features
        cv_exp_years = cv_data.get('total_experience_years', 0)
        job_exp_req = job_data.get('experience_years', 5)
        
        features.extend([
            cv_exp_years,
            cv_exp_years / job_exp_req if job_exp_req > 0 else 1,
            1 if cv_exp_years >= job_exp_req else 0
        ])
        
        # 3. Skills matching
        cv_skills = set([s.lower() for s in cv_data.get('skills', [])])
        job_skills = set([s.lower() for s in job_data.get('required_skills', [])])
        
        if job_skills:
            skill_match_ratio = len(cv_skills & job_skills) / len(job_skills)
            skill_coverage = len(cv_skills & job_skills) / len(cv_skills) if cv_skills else 0
        else:
            skill_match_ratio = 0
            skill_coverage = 0
        
        features.extend([
            skill_match_ratio,
            skill_coverage,
            len(cv_skills),
            len(cv_skills & job_skills)
        ])
        
        # 4. Education features
        cv_education = cv_data.get('education', [])
        education_levels = {
            'phd': 4, 'doctorate': 4,
            'master': 3, 'mba': 3,
            'bachelor': 2,
            'diploma': 1, 'certificate': 1
        }
        
        max_edu_level = 0
        for edu in cv_education:
            degree = edu.get('degree', '').lower()
            for key, level in education_levels.items():
                if key in degree:
                    max_edu_level = max(max_edu_level, level)
        
        features.append(max_edu_level)
        
        # 5. Certification features
        certifications = cv_data.get('certifications', [])
        features.append(len(certifications))
        
        # 6. Industry-specific keywords
        industry_keywords = [
            'leadership', 'strategic', 'management', 'stakeholder',
            'executive', 'director', 'budget', 'team', 'growth',
            'transformation', 'innovation', 'revenue', 'operations'
        ]
        
        keyword_count = sum(1 for kw in industry_keywords if kw in cv_text.lower())
        features.append(keyword_count / len(industry_keywords))
        
        # 7. Communication indicators
        comm_indicators = ['presented', 'negotiated', 'collaborated', 'communicated']
        comm_score = sum(1 for ind in comm_indicators if ind in cv_text.lower())
        features.append(comm_score / len(comm_indicators))
        
        # 8. Achievement indicators
        achievement_patterns = [
            r'increased.*by.*\d+%',
            r'reduced.*by.*\d+%',
            r'achieved.*\d+%',
            r'delivered.*\$\d+',
            r'managed.*\$\d+'
        ]
        achievement_count = sum(1 for pattern in achievement_patterns 
                              if re.search(pattern, cv_text, re.IGNORECASE))
        features.append(achievement_count)
        
        # 9. Language proficiency
        languages = cv_data.get('languages', [])
        features.append(len(languages))
        
        # 10. Location match
        cv_location = cv_data.get('contact_info', {}).get('location', '').lower()
        job_location = job_data.get('location', '').lower()
        location_match = 1 if job_location in cv_location or cv_location in job_location else 0
        features.append(location_match)
        
        return np.array(features)
    
    def extract_bert_features(self, cv_text: str, job_text: str) -> np.ndarray:
        """Extract BERT embeddings for semantic understanding"""
        if self.bert_extractor is None:
            self.bert_extractor = BERTFeatureExtractor()
        
        # Get embeddings for CV and job description
        cv_embedding = self.bert_extractor.extract_embeddings(cv_text[:512])
        job_embedding = self.bert_extractor.extract_embeddings(job_text[:512])
        
        # Compute similarity and difference features
        similarity = np.dot(cv_embedding, job_embedding) / (
            np.linalg.norm(cv_embedding) * np.linalg.norm(job_embedding)
        )
        
        # Return reduced features
        return np.array([similarity])
    
    def prepare_training_data(self, cvs_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from processed CVs"""
        X = []
        y = []
        
        stage_mapping = {
            'REJECT': 0,
            'SHORTLIST': 1,
            'INTERVIEW': 2,
            'ACCEPT': 3
        }
        
        for idx, row in cvs_df.iterrows():
            # Extract features
            cv_data = {
                'raw_text': row.get('raw_text', ''),
                'total_experience_years': row.get('total_experience_years', 0),
                'skills': row.get('skills', []),
                'education': row.get('education', []),
                'certifications': row.get('certifications', []),
                'languages': row.get('languages', []),
                'contact_info': row.get('contact_info', {})
            }
            
            job_data = {
                'description': row.get('job_description', ''),
                'experience_years': 5,  # Default or extract from job description
                'required_skills': [],  # Extract from job description
                'location': 'Darwin'  # Extract from job description
            }
            
            features = self.extract_advanced_features(cv_data, job_data)
            X.append(features)
            y.append(stage_mapping[row['stage']])
        
        return np.array(X), np.array(y)
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train ensemble of models"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Determine appropriate cv value based on sample size
        n_samples = len(X_train)
        cv_folds = min(5, n_samples) if n_samples >= 2 else None
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation score only if we have enough samples
            if cv_folds and cv_folds >= 2:
                scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                           epochs: int = 100, batch_size: int = 32):
        """Train custom neural network"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(self.scaler.transform(X_train))
        y_tensor = torch.LongTensor(y_train)
        
        # Initialize network
        input_size = X_train.shape[1]
        self.neural_net = RecruitmentNeuralNetwork(
            input_size=input_size,
            hidden_sizes=[128, 64, 32],
            num_classes=4
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                # Forward pass
                outputs = self.neural_net(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using ensemble voting"""
        # Check if scaler is fitted
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            X_scaled = X
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            # Check if model is fitted
            if hasattr(model, 'n_features_in_'):
                pred = model.predict(X_scaled)
                prob = model.predict_proba(X_scaled)
                predictions.append(pred)
                probabilities.append(prob)
            else:
                pass
        
        # Check if any models were fitted
        if not predictions:
            # Return default predictions
            return np.array([0]), np.array([[0.25, 0.25, 0.25, 0.25]])
        
        # Ensemble voting
        predictions = np.array(predictions)
        ensemble_pred = np.array([
            np.bincount(predictions[:, i]).argmax() 
            for i in range(predictions.shape[1])
        ])
        
        # Average probabilities
        ensemble_prob = np.mean(probabilities, axis=0)
        
        return ensemble_pred, ensemble_prob
    
    def explain_decision(self, cv_data: Dict, job_data: Dict, prediction: int) -> Dict:
        """Explain the recruitment decision"""
        features = self.extract_advanced_features(cv_data, job_data)
        
        stage_names = {0: 'REJECT', 1: 'SHORTLIST', 2: 'INTERVIEW', 3: 'ACCEPT'}
        
        explanation = {
            'decision': stage_names[prediction],
            'reasons': []
        }
        
        # Feature importance from Random Forest
        if hasattr(self.models['rf'], 'feature_importances_'):
            importances = self.models['rf'].feature_importances_
            feature_names = [
                'Text Similarity', 'Experience Years', 'Experience Ratio',
                'Meets Experience', 'Skill Match Ratio', 'Skill Coverage',
                'Total Skills', 'Matching Skills', 'Education Level',
                'Certifications', 'Industry Keywords', 'Communication Score',
                'Achievements', 'Languages', 'Location Match'
            ]
            
            # Get top contributing features
            top_indices = np.argsort(importances)[-5:][::-1]
            
            for idx in top_indices:
                if idx < len(feature_names):
                    feature_value = features[idx]
                    importance = importances[idx]
                    
                    if importance > 0.05:  # Only significant features
                        explanation['reasons'].append({
                            'feature': feature_names[idx],
                            'value': float(feature_value),
                            'importance': float(importance)
                        })
        
        # Add specific insights
        if cv_data.get('total_experience_years', 0) < job_data.get('experience_years', 5):
            explanation['reasons'].append({
                'feature': 'Experience Gap',
                'detail': f"Candidate has {cv_data.get('total_experience_years', 0)} years, job requires {job_data.get('experience_years', 5)}"
            })
        
        return explanation
    
    def save_model(self, path: str):
        """Save all models and preprocessors"""
        model_path = Path(path)
        model_path.mkdir(exist_ok=True)
        
        # Save sklearn models
        for name, model in self.models.items():
            joblib.dump(model, model_path / f"{name}_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, model_path / "scaler.pkl")
        
        # Save neural network
        if self.neural_net:
            torch.save(self.neural_net.state_dict(), model_path / "neural_net.pth")
    
    def load_model(self, path: str):
        """Load saved models"""
        model_path = Path(path)
        
        # Load sklearn models
        for name in self.models.keys():
            model_file = model_path / f"{name}_model.pkl"
            if model_file.exists():
                self.models[name] = joblib.load(model_file)
        
        # Load scaler
        scaler_file = model_path / "scaler.pkl"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)


class RecruitmentPipeline:
    """Complete recruitment pipeline"""
    
    def __init__(self, model_path: str = 'recruitment_model'):
        self.cv_processor = CVProcessor()
        self.model = AdvancedRecruitmentModel()
        
        # Try to load existing model
        try:
            self.model.load_model(model_path)
        except Exception as e:
            pass
        
    def process_new_application(self, cv_path: str, job_desc_path: str) -> Dict:
        """Process new job application"""
        # Process CV
        cv_data = self.cv_processor.process_cv(cv_path)
        
        # Process job description
        with open(job_desc_path, 'r', encoding='utf-8') as f:
            job_text = f.read()
        
        job_data = {
            'description': job_text,
            'experience_years': 5,  # Extract from job description
            'required_skills': [],  # Extract from job description
            'location': 'Darwin'  # Extract from job description
        }
        
        # Extract features
        features = self.model.extract_advanced_features(cv_data, job_data)
        
        # Make prediction
        prediction, probabilities = self.model.predict_ensemble(features.reshape(1, -1))
        
        # Get explanation
        explanation = self.model.explain_decision(cv_data, job_data, prediction[0])
        
        # Prepare result
        result = {
            'cv_path': cv_path,
            'prediction': explanation['decision'],
            'confidence': float(np.max(probabilities[0])),
            'probabilities': {
                'REJECT': float(probabilities[0][0]),
                'SHORTLIST': float(probabilities[0][1]),
                'INTERVIEW': float(probabilities[0][2]),
                'ACCEPT': float(probabilities[0][3])
            },
            'explanation': explanation,
            'candidate_summary': {
                'experience_years': cv_data.get('total_experience_years', 0),
                'skills_count': len(cv_data.get('skills', [])),
                'education_level': cv_data.get('education', []),
                'certifications': len(cv_data.get('certifications', []))
            }
        }
        
        return result


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RecruitmentPipeline()
    
    # Example usage
    # result = pipeline.process_new_application(
    #     "path/to/new_cv.pdf",
    #     "jobs/252_Chief Executive Officer 241212-01/job_description.txt"
    # )
    # print(json.dumps(result, indent=2))