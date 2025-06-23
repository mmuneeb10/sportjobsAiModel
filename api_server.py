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

from recruitment_ai_cli import RecruitmentAICLI
from cv_processor import CVProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the recruitment AI CLI
recruitment_ai = RecruitmentAICLI()

# Initialize CV processor
cv_processor = CVProcessor()

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

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)