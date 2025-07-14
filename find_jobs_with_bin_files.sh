#!/bin/bash

# Script to find jobs containing .bin files
# This script searches through the jobs directory and identifies which job folders contain .bin files

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Job Directories with .bin Files =====${NC}"
echo

# Initialize counters
total_jobs=0
jobs_with_bin=0
total_bin_files=0

# Check if jobs directory exists
if [ ! -d "jobs" ]; then
    echo -e "${RED}Error: 'jobs' directory not found!${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Create temporary file for storing results
temp_file=$(mktemp)

# Search for .bin files in each job directory
for job_dir in jobs/*/; do
    if [ -d "$job_dir" ]; then
        ((total_jobs++))
        
        # Count .bin files in this job directory
        bin_count=$(find "$job_dir" -name "*.bin" -type f 2>/dev/null | wc -l)
        
        if [ $bin_count -gt 0 ]; then
            ((jobs_with_bin++))
            ((total_bin_files+=$bin_count))
            
            # Extract job name from directory path
            job_name=$(basename "$job_dir")
            
            # Store result with count for sorting
            echo "$bin_count|$job_name|$job_dir" >> "$temp_file"
        fi
    fi
done

# Sort by number of .bin files (descending) and display results
if [ $jobs_with_bin -gt 0 ]; then
    echo -e "${GREEN}Jobs containing .bin files (sorted by count):${NC}"
    echo "----------------------------------------"
    
    # Sort and display results
    sort -t'|' -k1 -nr "$temp_file" | while IFS='|' read -r count name path; do
        printf "%-3d .bin files in: %s\n" "$count" "$name"
        
        # Optionally show subdirectory breakdown
        if [ "$1" == "--detailed" ] || [ "$1" == "-d" ]; then
            for subdir in ACCEPT INTERVIEW REJECT SHORTLIST; do
                subdir_count=$(find "$path$subdir" -name "*.bin" -type f 2>/dev/null | wc -l)
                if [ $subdir_count -gt 0 ]; then
                    printf "    └─ %-12s: %d .bin files\n" "$subdir" "$subdir_count"
                fi
            done
            echo
        fi
    done
else
    echo -e "${YELLOW}No jobs found with .bin files.${NC}"
fi

# Clean up temporary file
rm -f "$temp_file"

# Display summary
echo
echo -e "${BLUE}===== Summary =====${NC}"
echo "Total job directories scanned: $total_jobs"
echo "Jobs containing .bin files: $jobs_with_bin"
echo "Total .bin files found: $total_bin_files"

if [ $jobs_with_bin -gt 0 ]; then
    percentage=$(awk "BEGIN {printf \"%.1f\", ($jobs_with_bin/$total_jobs)*100}")
    echo "Percentage of jobs with .bin files: $percentage%"
fi

# Option to export results to file
if [ "$1" == "--export" ] || [ "$2" == "--export" ]; then
    output_file="jobs_with_bin_files_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Jobs with .bin files - Report generated on $(date)"
        echo "============================================="
        echo
        echo "Summary:"
        echo "- Total jobs scanned: $total_jobs"
        echo "- Jobs with .bin files: $jobs_with_bin"
        echo "- Total .bin files: $total_bin_files"
        echo
        echo "Job directories containing .bin files:"
        echo "--------------------------------------"
        
        # Re-run the search and save to file
        for job_dir in jobs/*/; do
            if [ -d "$job_dir" ]; then
                bin_count=$(find "$job_dir" -name "*.bin" -type f 2>/dev/null | wc -l)
                if [ $bin_count -gt 0 ]; then
                    job_name=$(basename "$job_dir")
                    echo "$job_name: $bin_count .bin files"
                fi
            fi
        done | sort -t: -k2 -nr
    } > "$output_file"
    
    echo
    echo -e "${GREEN}Results exported to: $output_file${NC}"
fi

# Show usage information
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --detailed, -d    Show breakdown by subdirectory (ACCEPT, INTERVIEW, etc.)"
    echo "  --export          Export results to a text file"
    echo "  --help, -h        Show this help message"
fi