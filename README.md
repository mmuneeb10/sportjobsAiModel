# Recruitment AI System

An advanced AI-powered recruitment system that learns from 20+ years of recruiting patterns to automate CV screening and candidate evaluation.

## Overview

This system analyzes CVs against job descriptions and predicts which recruitment stage a candidate should be placed in:
- **REJECT**: Not suitable for the position
- **SHORTLIST**: Potential candidate for initial screening
- **INTERVIEW**: Strong candidate for interviews
- **ACCEPT**: Top candidate for the position

## Features

- **Multi-format CV Processing**: Supports PDF, DOCX, DOC, and TXT files
- **Advanced Feature Extraction**: 
  - Text similarity analysis
  - Experience matching
  - Skills assessment
  - Education level evaluation
  - Certification recognition
  - Achievement detection
- **Ensemble Learning**: Combines multiple ML models for robust predictions
- **Explainable AI**: Provides reasons for each recruitment decision
- **Batch Processing**: Evaluate multiple CVs simultaneously
- **Historical Learning**: Trains on your historical recruitment data

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt --break-system-packages
```

## Project Structure

```
ai/
├── jobs/                          # Job folders with CVs
│   └── [Job_ID]/
│       ├── job_description.txt
│       ├── REJECT/               # Rejected CVs
│       ├── SHORTLIST/            # Shortlisted CVs
│       ├── INTERVIEW/            # Interview stage CVs
│       └── ACCEPT/               # Accepted CVs
├── recruitment_ai_system.py       # Basic AI model
├── cv_processor.py               # CV parsing and processing
├── advanced_recruitment_model.py  # Advanced ML models
├── train_recruitment_model.py    # Training pipeline
└── recruitment_ai_cli.py         # Command-line interface
```

## Usage

### 1. Training the Model

Train on your historical recruitment data:

```bash
python train_recruitment_model.py --jobs-dir jobs
```

This will:
- Load all CVs from your jobs folders
- Extract features from CVs and job descriptions
- Train multiple ML models
- Save the trained model to `recruitment_model/`

### 2. Evaluating a Single CV

```bash
python recruitment_ai_cli.py evaluate path/to/cv.pdf "jobs/252_Chief Executive Officer 241212-01/job_description.txt"
```

### 3. Batch Evaluation

Evaluate multiple CVs at once:

```bash
python recruitment_ai_cli.py batch folder/with/cvs "jobs/252_Chief Executive Officer 241212-01/job_description.txt" --output results.csv
```

### 4. Interactive Mode

For a guided experience:

```bash
python recruitment_ai_cli.py interactive
```

## How It Works

1. **CV Processing**: Extracts structured information from CVs including:
   - Contact information
   - Work experience and years
   - Education qualifications
   - Skills and certifications
   - Professional summary

2. **Feature Engineering**: Creates numerical features by comparing CV content with job requirements:
   - Text similarity scores
   - Experience match ratios
   - Skill coverage percentages
   - Education level matching
   - Industry keyword presence

3. **Multi-Model Ensemble**: Uses multiple algorithms for robust predictions:
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - Neural Networks

4. **Decision Explanation**: Provides interpretable reasons for each decision based on:
   - Feature importance
   - Missing requirements
   - Candidate strengths

## Adding Your Historical Data

To train the model on your 20+ years of recruitment data:

1. Organize CVs in the folder structure:
   ```
   jobs/
   └── [Job_Name_and_ID]/
       ├── job_description.txt
       ├── REJECT/     (place rejected CVs here)
       ├── SHORTLIST/  (place shortlisted CVs here)
       ├── INTERVIEW/  (place interview stage CVs here)
       └── ACCEPT/     (place accepted CVs here)
   ```

2. Run the training script:
   ```bash
   python train_recruitment_model.py
   ```

The model will learn your recruiting patterns and decision criteria.

## Model Performance

The system provides:
- Accuracy metrics for each recruitment stage
- Confusion matrix showing prediction patterns
- Feature importance analysis
- Confidence scores for each prediction

## Customization

You can customize the system by:
- Adding domain-specific keywords in `advanced_recruitment_model.py`
- Adjusting feature weights
- Adding new feature extractors
- Modifying the stage definitions

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

## Note

This system is designed to assist recruiters, not replace them. Always review AI recommendations and use professional judgment for final decisions.