#!/usr/bin/env python3
"""
REST API server for the Recruitment AI System
Provides POST endpoint to evaluate CVs against job descriptions via URLs
"""

from flask import Flask, request, jsonify
import requests
import tempfile
import os
from pathlib import Path
import logging
from urllib.parse import urlparse
import pandas as pd
import numpy as np

from recruitment_ai_cli import RecruitmentAICLI
from cv_processor import CVProcessor
from model1_binary_classifier import Model1BinaryClassifier
from model2_three_class_classifier import Model2ThreeClassClassifier
from advanced_recruitment_model import AdvancedRecruitmentModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the recruitment AI CLI
recruitment_ai = RecruitmentAICLI()

# Initialize CV processor
cv_processor = CVProcessor()

# Initialize binary models
model1_binary = Model1BinaryClassifier()
model2_three_class = Model2ThreeClassClassifier()
feature_extractor = AdvancedRecruitmentModel()

# Load trained models
try:
    model1_binary.load_model('model1_binary')
    model2_three_class.load_model('model2_three_class')
    logger.info("Binary models loaded successfully")
except Exception as e:
    logger.warning(f"Could not load binary models: {e}")
    model1_binary = None
    model2_three_class = None

def download_file_from_url(url: str, suffix: str = None) -> str:
    """
    Download a file from URL to a temporary file
    
    Args:
        url: URL to download from
        suffix: File extension for the temporary file
    
    Returns:
        Path to the temporary file
    """
    try:
        # Make the request
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix or os.path.splitext(urlparse(url).path)[1]
        )
        
        # Write content to temp file
        temp_file.write(response.content)
        temp_file.close()
        
        logger.info(f"Downloaded file from {url} to {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {e}")
        raise

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary file {file_path}: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "recruitment-ai-api"})

@app.route('/evaluate', methods=['POST'])
def evaluate_cv():
    """
    Evaluate a CV against a job description
    
    Expected JSON payload:
    {
        "cv_url": "https://example.com/cv.pdf",
        "job_description_url": "https://example.com/job.txt"
    }
    
    Returns:
    JSON response with evaluation results
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        # Check required fields
        if 'cv_url' not in data or 'job_description_url' not in data:
            return jsonify({
                "error": "Missing required fields: cv_url and job_description_url"
            }), 400
        
        cv_url = data['cv_url']
        job_desc_url = data['job_description_url']
        
        logger.info(f"Processing evaluation request - CV: {cv_url}, Job: {job_desc_url}")
        
        # Download files to temporary locations
        cv_temp_path = None
        job_temp_path = None
        
        try:
            # Download CV file
            cv_temp_path = download_file_from_url(cv_url)
            
            # Download job description file
            job_temp_path = download_file_from_url(job_desc_url)
            
            # Evaluate the CV
            result = recruitment_ai.evaluate_single_cv(cv_temp_path, job_temp_path)
            
            if result is None:
                return jsonify({"error": "Failed to evaluate CV"}), 500
            
            # Return the result
            return jsonify({
                "success": True,
                "evaluation": result
            })
            
        finally:
            # Clean up temporary files
            if cv_temp_path:
                cleanup_temp_file(cv_temp_path)
            if job_temp_path:
                cleanup_temp_file(job_temp_path)
                
    except requests.RequestException as e:
        logger.error(f"Error downloading files: {e}")
        return jsonify({"error": f"Failed to download files: {str(e)}"}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/extract-name', methods=['POST'])
def extract_name_from_cv():
    """
    Extract candidate name from CV file
    
    Expected JSON payload:
    {
        "cv_url": "https://example.com/cv.pdf"
    }
    
    Returns:
    {
        "success": true,
        "name": "John Doe",
        "confidence": "high"
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        # Check required field
        if 'cv_url' not in data:
            return jsonify({
                "error": "Missing required field: cv_url"
            }), 400
        
        cv_url = data['cv_url']
        
        logger.info(f"Processing name extraction request for CV: {cv_url}")
        
        # Download file to temporary location
        cv_temp_path = None
        
        try:
            # Download CV file
            cv_temp_path = download_file_from_url(cv_url)
            
            # Process the CV to extract data
            cv_data = cv_processor.process_cv(cv_temp_path)
            
            # Get raw text
            raw_text = cv_data.get('raw_text', '')
            
            if not raw_text:
                return jsonify({
                    "success": False,
                    "error": "Could not extract text from CV"
                }), 400
            
            # Extract name using the dedicated method
            extracted_name = cv_processor.extract_name(raw_text)
            
            # Determine confidence level based on extraction
            confidence = "high" if extracted_name else "low"
            
            # If no name found with high confidence, try contact info
            if not extracted_name:
                contact_info = cv_data.get('contact_info', {})
                # Try to find name in contact info (if implemented there)
                if 'name' in contact_info:
                    extracted_name = contact_info['name']
                    confidence = "medium"
            
            # Log the result
            logger.info(f"Name extraction result: '{extracted_name}' with confidence: {confidence}")
            
            # Return the result
            return jsonify({
                "success": True,
                "name": extracted_name,
                "confidence": confidence
            })
            
        finally:
            # Clean up temporary file
            if cv_temp_path:
                cleanup_temp_file(cv_temp_path)
                
    except requests.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({"error": f"Failed to download file: {str(e)}"}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error during name extraction: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/evaluate_binary', methods=['POST'])
def evaluate_binary():
    """
    Evaluate a CV against a job description using two-stage binary classification
    
    Expected JSON payload:
    {
        "cv_url": "https://example.com/cv.pdf",
        "job_description_url": "https://example.com/job.txt"
    }
    
    Returns:
    JSON response with two-stage evaluation results
    """
    try:
        # Check if models are loaded
        if model1_binary is None or model2_three_class is None:
            return jsonify({
                "error": "Binary models not loaded. Please train the models first."
            }), 503
        
        # Check if models have been trained (have best_model_name)
        if not hasattr(model1_binary, 'best_model_name') or model1_binary.best_model_name is None:
            return jsonify({
                "error": "Model 1 (binary) not trained. Please train the models first."
            }), 503
            
        if not hasattr(model2_three_class, 'best_model_name') or model2_three_class.best_model_name is None:
            return jsonify({
                "error": "Model 2 (three-class) not trained. Please train the models first."
            }), 503
        
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        # Check required fields
        if 'cv_url' not in data or 'job_description_url' not in data:
            return jsonify({
                "error": "Missing required fields: cv_url and job_description_url"
            }), 400
        
        cv_url = data['cv_url']
        job_desc_url = data['job_description_url']
        
        logger.info(f"Processing binary evaluation request - CV: {cv_url}, Job: {job_desc_url}")
        
        # Download files to temporary locations
        cv_temp_path = None
        job_temp_path = None
        
        try:
            # Download CV file
            cv_temp_path = download_file_from_url(cv_url)
            
            # Download job description file
            job_temp_path = download_file_from_url(job_desc_url)
            
            # Process CV to extract data
            try:
                cv_data = cv_processor.process_cv(cv_temp_path)
                if not cv_data:
                    return jsonify({"error": "Failed to process CV file"}), 400
            except Exception as e:
                logger.error(f"Error processing CV: {e}")
                return jsonify({"error": f"CV processing failed: {str(e)}"}), 500
            
            # Read job description
            try:
                with open(job_temp_path, 'r', encoding='utf-8') as f:
                    job_description = f.read()
                if not job_description.strip():
                    return jsonify({"error": "Job description file is empty"}), 400
            except Exception as e:
                logger.error(f"Error reading job description: {e}")
                return jsonify({"error": f"Failed to read job description: {str(e)}"}), 500
            
            # Create a temporary DataFrame for feature extraction
            temp_df = pd.DataFrame([{
                'stage': 'UNKNOWN',  # Placeholder
                'raw_text': cv_data.get('raw_text', ''),
                'total_experience_years': cv_data.get('total_experience_years', 0),
                'skills': cv_data.get('skills', []),
                'education': cv_data.get('education', []),
                'certifications': cv_data.get('certifications', []),
                'languages': cv_data.get('languages', []),
                'contact_info': cv_data.get('contact_info', {}),
                'job_description': job_description
            }])
            
            # Extract features using the advanced model
            try:
                X, _ = feature_extractor.prepare_training_data(temp_df)
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                # Fallback to basic feature extraction using extract_advanced_features
                try:
                    job_data = {
                        'description': job_description,
                        'experience_years': 5,  # Default
                        'required_skills': [],  # Could be extracted from job description
                        'location': 'Darwin'  # Default
                    }
                    features = feature_extractor.extract_advanced_features(cv_data, job_data)
                    X = np.array([features])  # Make it 2D array for single sample
                except Exception as e2:
                    logger.error(f"Fallback feature extraction also failed: {e2}")
                    return jsonify({"error": f"Feature extraction failed: {str(e2)}"}), 500
            
            # Stage 1: Binary classification (REJECT vs Others)
            try:
                binary_prediction, binary_probability = model1_binary.predict(X)
            except Exception as e:
                logger.error(f"Error in binary prediction: {e}")
                return jsonify({"error": f"Binary prediction failed: {str(e)}"}), 500
            
            stage1_result = {
                'prediction': 'Others' if binary_prediction[0] == 1 else 'REJECT',
                'probability': float(binary_probability[0]),
                'confidence': 'High' if binary_probability[0] > 0.8 or binary_probability[0] < 0.2 else 'Medium'
            }
            
            # Stage 2: If not rejected, classify into SHORTLIST/INTERVIEW/ACCEPT
            final_prediction = None
            stage2_result = None
            
            if binary_prediction[0] == 1:  # If suitable (not rejected)
                try:
                    three_class_prediction, three_class_probabilities = model2_three_class.predict(X)
                except Exception as e:
                    logger.error(f"Error in three-class prediction: {e}")
                    return jsonify({"error": f"Three-class prediction failed: {str(e)}"}), 500
                
                # Map prediction to stage name
                stage_names = ['SHORTLIST', 'INTERVIEW', 'ACCEPT']
                final_prediction = stage_names[three_class_prediction[0]]
                
                stage2_result = {
                    'prediction': final_prediction,
                    'probabilities': {
                        'SHORTLIST': float(three_class_probabilities[0][0]),
                        'INTERVIEW': float(three_class_probabilities[0][1]),
                        'ACCEPT': float(three_class_probabilities[0][2])
                    },
                    'confidence': 'High' if max(three_class_probabilities[0]) > 0.6 else 'Medium'
                }
            else:
                final_prediction = 'REJECT'
            
            # Create probabilities dict to match /evaluate format
            if final_prediction == 'REJECT':
                # If rejected, set REJECT prob high and others low
                probabilities = {
                    'REJECT': float(1.0 - binary_probability[0]),  # High probability for REJECT
                    'SHORTLIST': 0.0,
                    'INTERVIEW': 0.0,
                    'ACCEPT': 0.0
                }
                confidence = float(1.0 - binary_probability[0])
            else:
                # If not rejected, use stage2 probabilities
                probabilities = {
                    'REJECT': float(1.0 - binary_probability[0]),  # Low probability for REJECT
                    'SHORTLIST': float(three_class_probabilities[0][0]),
                    'INTERVIEW': float(three_class_probabilities[0][1]),
                    'ACCEPT': float(three_class_probabilities[0][2])
                }
                confidence = float(max(three_class_probabilities[0]))
            
            # Match exact format from /evaluate endpoint
            evaluation_result = {
                'cv_path': cv_url,  # Use URL since we don't have local path
                'prediction': final_prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'explanation': {
                    'decision': final_prediction,
                    'reasoning': f"Two-stage binary classification: Stage 1 ({'REJECT' if final_prediction == 'REJECT' else 'Others'}), Stage 2 ({final_prediction if final_prediction != 'REJECT' else 'N/A'})"
                },
                'candidate_summary': {
                    'experience_years': cv_data.get('total_experience_years', 0),
                    'skills_count': len(cv_data.get('skills', [])),
                    'education_level': cv_data.get('education', []),
                    'certifications': len(cv_data.get('certifications', []))
                }
            }
            
            logger.info(f"Binary evaluation completed. Final prediction: {final_prediction}")
            
            # Return in exact same format as /evaluate endpoint
            return jsonify({
                "success": True,
                "evaluation": evaluation_result
            })
            
        finally:
            # Clean up temporary files
            if cv_temp_path:
                cleanup_temp_file(cv_temp_path)
            if job_temp_path:
                cleanup_temp_file(job_temp_path)
                
    except requests.RequestException as e:
        logger.error(f"Error downloading files: {e}")
        return jsonify({"error": f"Failed to download files: {str(e)}"}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error in binary evaluation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)