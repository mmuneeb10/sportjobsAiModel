#!/bin/bash

# Script to delete all .bin files from jobs directory
# This will search and remove all files with .bin extension

echo "Starting cleanup of .bin files..."
echo "================================"

# Counter for deleted files
deleted_count=0

# Find and delete all .bin files in jobs directory
find jobs -name "*.bin" -type f | while read -r file; do
    echo "Deleting: $file"
    rm -f "$file"
    ((deleted_count++))
done

# Count remaining .bin files to verify
remaining=$(find jobs -name "*.bin" -type f | wc -l)

echo "================================"
echo "Cleanup completed!"
echo "Files deleted: $(find jobs -name "*.bin" -type f -delete -print | wc -l)"
echo "Remaining .bin files: $remaining"

# Make the script exit successfully
exit 0