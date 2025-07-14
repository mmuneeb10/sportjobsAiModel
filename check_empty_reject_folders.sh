#!/bin/bash

# Script to find job folders where REJECT folder has no CVs
# This helps identify jobs that might need review

echo "Checking for jobs with empty REJECT folders..."
echo "============================================="

empty_reject_jobs=()
total_jobs=0

# Find all job folders (those containing job_description.txt)
for job_dir in jobs/*/; do
    if [ -f "$job_dir/job_description.txt" ]; then
        total_jobs=$((total_jobs + 1))
        job_name=$(basename "$job_dir")
        
        # Check if REJECT folder exists
        if [ -d "$job_dir/REJECT" ]; then
            # Count files in REJECT folder
            file_count=$(find "$job_dir/REJECT" -type f 2>/dev/null | wc -l)
            
            if [ "$file_count" -eq 0 ]; then
                empty_reject_jobs+=("$job_name")
                echo "❌ No CVs in REJECT: $job_name"
            fi
        else
            # REJECT folder doesn't exist
            empty_reject_jobs+=("$job_name")
            echo "❌ No REJECT folder: $job_name"
        fi
    fi
done

echo ""
echo "============================================="
echo "SUMMARY:"
echo "Total jobs checked: $total_jobs"
echo "Jobs with empty REJECT folders: ${#empty_reject_jobs[@]}"
echo ""

if [ ${#empty_reject_jobs[@]} -gt 0 ]; then
    echo "List of jobs with empty REJECT folders:"
    echo "---------------------------------------"
    for job in "${empty_reject_jobs[@]}"; do
        echo "- $job"
    done
else
    echo "✅ All jobs have CVs in their REJECT folders!"
fi