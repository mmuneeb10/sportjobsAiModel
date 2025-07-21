#!/usr/bin/env python3
"""
Script to consolidate CVs from ACCEPT, PLACED, and INTERVIEW folders into SHORTLIST folder
for all jobs in the jobs directory, and remove the empty folders after moving files.
This version runs automatically without confirmation prompt.
"""

import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def consolidate_folders_to_shortlist(jobs_dir: str):
    """
    Move all files from ACCEPT, PLACED, and INTERVIEW folders to SHORTLIST folder
    for each job directory and remove the empty folders.
    
    Args:
        jobs_dir: Path to the jobs directory
    """
    jobs_path = Path(jobs_dir)
    
    if not jobs_path.exists():
        logger.error(f"Jobs directory not found: {jobs_dir}")
        return
    
    # Folders to consolidate into SHORTLIST
    source_folders = ['ACCEPT', 'PLACED', 'INTERVIEW']
    target_folder = 'SHORTLIST'
    
    # Statistics
    total_jobs = 0
    total_files_moved = 0
    total_folders_removed = 0
    
    # Process each job directory
    for job_dir in jobs_path.iterdir():
        if not job_dir.is_dir():
            continue
            
        total_jobs += 1
        job_name = job_dir.name
        logger.info(f"Processing job: {job_name}")
        
        # Create SHORTLIST folder if it doesn't exist
        shortlist_path = job_dir / target_folder
        if not shortlist_path.exists():
            shortlist_path.mkdir()
            logger.info(f"  Created {target_folder} folder")
        
        # Process each source folder
        for folder_name in source_folders:
            source_path = job_dir / folder_name
            
            if source_path.exists() and source_path.is_dir():
                # Count files in this folder
                files_in_folder = list(source_path.iterdir())
                file_count = len([f for f in files_in_folder if f.is_file()])
                
                if file_count > 0:
                    logger.info(f"  Moving {file_count} files from {folder_name} to {target_folder}")
                    
                    # Move each file
                    for file_path in source_path.iterdir():
                        if file_path.is_file():
                            # Construct destination path
                            dest_path = shortlist_path / file_path.name
                            
                            # Handle duplicate filenames
                            if dest_path.exists():
                                # Add a suffix to avoid overwriting
                                base_name = file_path.stem
                                extension = file_path.suffix
                                counter = 1
                                while dest_path.exists():
                                    new_name = f"{base_name}_{counter}{extension}"
                                    dest_path = shortlist_path / new_name
                                    counter += 1
                                logger.warning(f"    Duplicate file found, renaming to: {dest_path.name}")
                            
                            # Move the file
                            shutil.move(str(file_path), str(dest_path))
                            total_files_moved += 1
                
                # Remove the empty folder
                try:
                    source_path.rmdir()
                    logger.info(f"  Removed empty {folder_name} folder")
                    total_folders_removed += 1
                except OSError as e:
                    logger.warning(f"  Could not remove {folder_name} folder: {e}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CONSOLIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total jobs processed: {total_jobs}")
    logger.info(f"Total files moved: {total_files_moved}")
    logger.info(f"Total folders removed: {total_folders_removed}")
    logger.info("="*60)

def main():
    """Main function"""
    # Define the jobs directory path
    jobs_dir = "/Users/user/Documents/Workspace/Sportsjob model/sportjobsAiModel/jobs"
    
    logger.info("Starting consolidation process...")
    logger.info(f"Processing jobs in: {jobs_dir}")
    
    consolidate_folders_to_shortlist(jobs_dir)
    
    logger.info("\nConsolidation complete!")

if __name__ == "__main__":
    main()