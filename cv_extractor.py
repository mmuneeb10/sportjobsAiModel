#!/usr/bin/env python3
"""
CV Extractor Script
This script extracts CVs from database URLs and organizes them into folders
based on job descriptions and candidate statuses.
"""

# Import necessary libraries for database connection, file operations, and HTTP requests
import os  # For file and directory operations
import sys  # For system-specific parameters and functions
import requests  # For downloading files from URLs
import psycopg2  # For PostgreSQL database connection
import psycopg2.extras  # For DictCursor to get results as dictionaries
import logging  # For logging errors and information
from datetime import datetime  # For timestamp operations
from urllib.parse import urlparse  # For parsing URLs
from pathlib import Path  # For path operations
import shutil  # For file operations like copying
from typing import List, Dict, Tuple, Optional  # For type hints

# Configure logging to track the script's execution
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO to capture important events
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log messages
    handlers=[
        logging.FileHandler('cv_extractor.log'),  # Log to file
        logging.StreamHandler(sys.stdout)  # Also log to console
    ]
)
logger = logging.getLogger(__name__)  # Create logger instance for this module


class CVExtractor:
    """
    Main class for extracting CVs from database and organizing them into folders
    """
    
    def __init__(self, db_config: dict):
        """
        Initialize the CV Extractor with database configuration
        
        Args:
            db_config (dict): Database configuration containing host, user, password, database name
        """
        # Store database configuration for later use
        self.db_config = db_config
        
        # Initialize database connection as None (will be established when needed)
        self.connection = None
        
        # Define the base directory where CVs will be stored
        self.base_cv_folder = "CVS"
        
        # Create the base CV folder if it doesn't exist
        self._ensure_directory_exists(self.base_cv_folder)
        
    def _ensure_directory_exists(self, directory_path: str) -> None:
        """
        Create directory if it doesn't exist
        
        Args:
            directory_path (str): Path of the directory to create
        """
        # Use Path object for cross-platform compatibility
        path = Path(directory_path)
        
        # Create directory and all parent directories if they don't exist
        path.mkdir(parents=True, exist_ok=True)
        
        # Log the directory creation
        logger.info(f"Ensured directory exists: {directory_path}")
    
    def connect_to_database(self) -> None:
        """
        Establish connection to the database using the provided configuration
        """
        try:
            # Attempt to create database connection using psycopg2 for PostgreSQL
            self.connection = psycopg2.connect(
                host=self.db_config['host'],  # Database host (e.g., 'localhost' or IP address)
                user=self.db_config['user'],  # Database username
                password=self.db_config['password'],  # Database password
                database=self.db_config['database'],  # Database name
                port=self.db_config.get('port', 5432)  # Database port (default PostgreSQL port)
            )
            # Set autocommit to True for read operations
            self.connection.autocommit = True
            # Log successful connection
            logger.info("Successfully connected to database")
        except Exception as e:
            # Log error and re-raise exception
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def close_database_connection(self) -> None:
        """
        Close the database connection if it exists
        """
        # Check if connection exists and is open
        if self.connection:
            # Close the connection
            self.connection.close()
            # Log the closure
            logger.info("Database connection closed")
    
    def fetch_job_description(self, job_description_id: int) -> Optional[Dict]:
        """
        Fetch job description details from the BullhornJob table
        
        Args:
            job_description_id (int): The bullhornId of the job description to fetch
            
        Returns:
            dict: Job description details or None if not found
        """
        try:
            # Create a cursor to execute database queries with results as dictionaries
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # SQL query to fetch job description from BullhornJob table
                query = """
                SELECT 
                    "bullhornId",  -- Bullhorn Job ID
                    "title",  -- Job title
                    "publicDescription",  -- Public job description
                    "description",  -- Full job description
                    "clientCorporationName",  -- Client corporation name
                    "dateAdded"  -- Date when job was added
                FROM "BullhornJob"  -- Bullhorn job table
                WHERE "bullhornId" = %s  -- Filter by bullhorn job ID
                """
                
                # Execute query with parameter to prevent SQL injection
                cursor.execute(query, (job_description_id,))
                
                # Fetch the result
                result = cursor.fetchone()
                
                # Log the result
                if result:
                    logger.info(f"Found job description: {result['title']}")
                else:
                    logger.warning(f"No job description found with bullhornId: {job_description_id}")
                
                return result
                
        except Exception as e:
            # Log error and return None
            logger.error(f"Error fetching job description: {str(e)}")
            return None
    
    def fetch_candidate_submissions(self, job_description_id: int) -> List[Dict]:
        """
        Fetch all candidate submissions from BullhornJobSubmission table for a job
        
        Args:
            job_description_id (int): The bullhornJobId to fetch submissions for
            
        Returns:
            list: List of candidate submission records with status
        """
        try:
            # Create a cursor to execute database queries with results as dictionaries
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # SQL query to fetch candidate submissions from BullhornJobSubmission
                # Join with BullhornCandidate to get candidate name
                query = """
                SELECT 
                    s."id" as submission_id,  -- Submission ID
                    s."bullhornCandidateId",  -- Candidate ID (foreign key)
                    CONCAT(c."firstName", ' ', c."lastName") as candidateName,  -- Candidate name from BullhornCandidate
                    s."status",  -- Submission status (submitted, interviewed, hired, etc.)
                    s."dateAdded",  -- Date when submission was added
                    s."bullhornJobId"  -- Job ID (foreign key)
                FROM "BullhornJobSubmission" s  -- Bullhorn job submission table
                LEFT JOIN "BullhornCandidate" c ON s."bullhornCandidateId" = c."bullhornId"  -- Join to get candidate details
                WHERE s."bullhornJobId" = %s  -- Filter by job ID
                """
                
                # Execute query with parameter
                cursor.execute(query, (job_description_id,))
                
                # Fetch all results
                results = cursor.fetchall()
                
                # Log the number of submissions found
                logger.info(f"Found {len(results)} candidate submissions for job ID: {job_description_id}")
                
                return results
                
        except Exception as e:
            # Log error and return empty list
            logger.error(f"Error fetching candidate submissions: {str(e)}")
            return []
    
    def fetch_candidate_cv_url(self, bullhorn_candidate_id: int) -> Optional[str]:
        """
        Fetch the CV URL for a candidate from BullhornFile table
        
        Args:
            bullhorn_candidate_id (int): The bullhornCandidateId from submission
            
        Returns:
            str: CV URL or None if not found
        """
        try:
            # Create a cursor to execute database queries with results as dictionaries
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # SQL query to fetch CV URL from BullhornFile table
                query = """
                SELECT 
                    "id" as file_id,  -- File ID
                    "s3Url",  -- S3 URL of the CV file
                    "fileName",  -- Name of the file
                    "contentType",  -- Content type of file (application/pdf, etc.)
                    "dateAdded"  -- Date when file was added
                FROM "BullhornFile"  -- Bullhorn file table
                WHERE "bullhornCandidateId" = %s  -- Filter by candidate ID
                AND "contentType" IN ('application/pdf', 'application/msword', 
                                   'application/vnd.openxmlformats-officedocument.wordprocessingml.document')  -- Only get CV files
                ORDER BY "dateAdded" DESC  -- Get the most recent file
                LIMIT 1  -- Only get the latest CV
                """
                
                # Execute query with parameter
                cursor.execute(query, (bullhorn_candidate_id,))
                
                # Fetch the result
                result = cursor.fetchone()
                
                # Return CV URL if found
                if result:
                    logger.info(f"Found CV file for candidate {bullhorn_candidate_id}: {result['fileName']}")
                    return result['s3Url']
                else:
                    logger.warning(f"No CV file found for candidate {bullhorn_candidate_id}")
                    return None
                    
        except Exception as e:
            # Log error and return None
            logger.error(f"Error fetching candidate CV URL: {str(e)}")
            return None
    
    def download_cv_from_url(self, cv_url: str, destination_path: str) -> bool:
        """
        Download CV PDF from URL and save to specified path
        
        Args:
            cv_url (str): URL of the CV file
            destination_path (str): Path where the CV should be saved
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Log the download attempt
            logger.info(f"Downloading CV from: {cv_url}")
            
            # Set headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Send GET request to download the file
            response = requests.get(
                cv_url, 
                headers=headers, 
                timeout=30,  # 30 second timeout
                stream=True  # Stream the download for large files
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Write the content to file in chunks to handle large files
            with open(destination_path, 'wb') as file:
                # Download in 8KB chunks
                for chunk in response.iter_content(chunk_size=8192):
                    # Filter out keep-alive chunks
                    if chunk:
                        file.write(chunk)
            
            # Log successful download
            logger.info(f"Successfully downloaded CV to: {destination_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            # Log HTTP-related errors
            logger.error(f"Error downloading CV from {cv_url}: {str(e)}")
            return False
        except Exception as e:
            # Log any other errors
            logger.error(f"Unexpected error downloading CV: {str(e)}")
            return False
    
    def save_job_description_as_txt(self, job_description: Dict, job_folder_path: str) -> None:
        """
        Save the job description as a text file in the job folder
        
        Args:
            job_description (dict): Job description details containing publicDescription or description
            job_folder_path (str): Path to the job folder
        """
        try:
            # Extract the job description data (prefer publicDescription, fallback to description)
            job_desc_text = job_description.get('publicDescription', '') or job_description.get('description', '')
            
            # If no job description found, log warning
            if not job_desc_text:
                logger.warning(f"No publicDescription or description found for job ID: {job_description['bullhornId']}")
                return
            
            # Create filename for the job description
            jd_filename = "job_description.txt"
            jd_path = os.path.join(job_folder_path, jd_filename)
            
            # Write the job description to text file
            with open(jd_path, 'w', encoding='utf-8') as file:
                file.write(f"Job Title: {job_description.get('title', 'N/A')}\n")
                file.write(f"Company: {job_description.get('clientCorporationName', 'N/A')}\n")
                file.write(f"Job ID: {job_description.get('bullhornId', 'N/A')}\n")
                file.write(f"Date Added: {job_description.get('dateAdded', 'N/A')}\n")
                file.write("-" * 50 + "\n\n")
                file.write("JOB DESCRIPTION:\n\n")
                file.write(job_desc_text)
            
            # Log successful save
            logger.info(f"Saved job description to: {jd_path}")
            
        except Exception as e:
            # Log error
            logger.error(f"Error saving job description: {str(e)}")
    
    def create_folder_structure(self, job_description: Dict, status: str = None) -> Tuple[str, str]:
        """
        Create the folder structure for organizing CVs
        
        Args:
            job_description (dict): Job description details
            status (str, optional): Candidate status for creating status subfolder
            
        Returns:
            tuple: (job_folder_path, status_folder_path) - Path to job folder and status folder
        """
        # Clean the job title to make it filesystem-friendly
        # Remove special characters that might cause issues in folder names
        job_title = job_description.get('title', 'Untitled').replace('/', '_').replace('\\', '_')
        job_title = ''.join(c for c in job_title if c.isalnum() or c in (' ', '-', '_'))
        
        # Create folder name with job ID and title for uniqueness
        job_folder_name = f"{job_description['bullhornId']}_{job_title}"
        
        # Build the job folder path
        # Structure: CVS/JobID_JobTitle/
        job_folder_path = os.path.join(
            self.base_cv_folder,  # Base CVS folder
            job_folder_name  # Job-specific folder
        )
        
        # Create the job directory
        self._ensure_directory_exists(job_folder_path)
        
        # If status is provided, create status subfolder
        status_folder_path = job_folder_path
        if status:
            # Build the status folder path
            # Structure: CVS/JobID_JobTitle/Status/
            status_folder_path = os.path.join(job_folder_path, status)
            
            # Create the status directory
            self._ensure_directory_exists(status_folder_path)
            
            # Log the folder creation
            logger.info(f"Created folder structure: {status_folder_path}")
        
        return job_folder_path, status_folder_path
    
    def process_job_cvs(self, job_description_id: int) -> None:
        """
        Main method to process all CVs for a given job description
        
        Args:
            job_description_id (int): The bullhornJobId to process
        """
        try:
            # Log the start of processing
            logger.info(f"Starting CV extraction for Bullhorn Job ID: {job_description_id}")
            
            # Step 1: Fetch job description details from BullhornJob table
            job_description = self.fetch_job_description(job_description_id)
            if not job_description:
                logger.error(f"Job description with bullhornId {job_description_id} not found in BullhornJob table")
                return
            
            # Step 2: Create main job folder and save job description as text
            job_folder_path, _ = self.create_folder_structure(job_description)
            
            # Save the job description as a text file
            self.save_job_description_as_txt(job_description, job_folder_path)
            
            # Step 3: Fetch all candidate submissions for this job from BullhornJobSubmission
            candidate_submissions = self.fetch_candidate_submissions(job_description_id)
            if not candidate_submissions:
                logger.warning(f"No candidate submissions found for job ID: {job_description_id}")
                return
            
            # Initialize counters for tracking success/failure
            success_count = 0
            failure_count = 0
            
            # Step 4: Process each candidate submission
            for submission in candidate_submissions:
                try:
                    # Extract submission information
                    submission_id = submission['submission_id']
                    bullhorn_candidate_id = submission['bullhornCandidateId']
                    candidate_name = submission.get('candidateName', f'Candidate_{bullhorn_candidate_id}')
                    status = submission.get('status', 'unknown')
                    
                    # Log processing of this candidate
                    logger.info(f"Processing candidate: {candidate_name} (ID: {bullhorn_candidate_id}, Status: {status})")
                    
                    # Step 5: Fetch CV URL from BullhornFile table using bullhornCandidateId
                    cv_url = self.fetch_candidate_cv_url(bullhorn_candidate_id)
                    
                    # Skip if CV URL is empty or None
                    if not cv_url:
                        logger.warning(f"No CV URL found for candidate {bullhorn_candidate_id} in BullhornFile table")
                        failure_count += 1
                        continue
                    
                    # Step 6: Create status-specific folder structure
                    _, status_folder = self.create_folder_structure(job_description, status)
                    
                    # Generate filename for the CV
                    # Clean candidate name for filesystem
                    clean_name = ''.join(c for c in candidate_name if c.isalnum() or c in (' ', '-', '_'))
                    # Create filename with candidate ID and name
                    cv_filename = f"{bullhorn_candidate_id}_{clean_name}.pdf"
                    # Full path for the CV file
                    cv_path = os.path.join(status_folder, cv_filename)
                    
                    # Step 7: Download CV from URL
                    if self.download_cv_from_url(cv_url, cv_path):
                        success_count += 1
                        logger.info(f"Successfully downloaded CV for candidate: {candidate_name}")
                    else:
                        failure_count += 1
                        logger.error(f"Failed to download CV for candidate: {candidate_name}")
                        
                except Exception as e:
                    # Log error for individual CV processing
                    failure_count += 1
                    logger.error(f"Error processing submission {submission.get('submission_id', 'unknown')}: {str(e)}")
            
            # Log summary of processing
            logger.info(f"CV extraction completed for Bullhorn Job ID {job_description_id}")
            logger.info(f"Total submissions: {len(candidate_submissions)}")
            logger.info(f"Successfully downloaded: {success_count}")
            logger.info(f"Failed downloads: {failure_count}")
            
        except Exception as e:
            # Log error for the entire job processing
            logger.error(f"Error processing job {job_description_id}: {str(e)}")
            raise


def main():
    """
    Main function to execute the CV extraction process
    """
    # ================== CONFIGURATION SECTION ==================
    # IMPORTANT: Update these values with your actual database credentials
    db_config = {
        'host': 'sport-jobs-db.covy88q4ili5.us-east-1.rds.amazonaws.com',  # Your database host (e.g., 'localhost', '192.168.1.100', 'db.example.com')
        'user': 'postgres',  # Your database username
        'password': 'mAIFL7HOXgthMiZbygQS',  # Your database password
        'database': 'sport_jobs'  # Your database name containing Bullhorn tables
    }
    
    # ================== JOB ID CONFIGURATION ==================
    # Option 1: Single job processing
    # Replace 123 with the actual bullhornJobId you want to process
    # This need to be the bullhornId for the bullHornJob not the id
    job_description_id = 261  # Example: 456789
    
    # Option 2: Multiple jobs processing (uncomment to use)
    # job_ids = [123, 124, 125]  # List of bullhornJobIds to process
    
    # Option 3: Command line argument (uncomment to use)
    # import sys
    # if len(sys.argv) > 1:
    #     job_description_id = int(sys.argv[1])
    # else:
    #     print("Usage: python cv_extractor.py <job_id>")
    #     return
    
    # ================== EXECUTION SECTION ==================
    # Create instance of CVExtractor with database configuration
    extractor = CVExtractor(db_config)
    
    try:
        # Connect to database
        logger.info("Connecting to database...")
        extractor.connect_to_database()
        
        # Process CVs for the specified job
        logger.info(f"Processing CVs for Bullhorn Job ID: {job_description_id}")
        extractor.process_job_cvs(job_description_id)
        
        # Option 2: Process multiple jobs (uncomment if using job_ids list)
        # for job_id in job_ids:
        #     logger.info(f"Processing CVs for Bullhorn Job ID: {job_id}")
        #     extractor.process_job_cvs(job_id)
        
        logger.info("CV extraction process completed successfully!")
        
    except Exception as e:
        # Log any errors that occur
        logger.error(f"Script execution failed: {str(e)}")
        raise  # Re-raise to see full traceback
        
    finally:
        # Always close database connection
        extractor.close_database_connection()


if __name__ == "__main__":
    """
    Entry point of the script - runs when script is executed directly
    """
    main()