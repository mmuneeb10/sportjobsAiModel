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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the recruitment AI CLI
recruitment_ai = RecruitmentAICLI()

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

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)