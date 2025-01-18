#!/bin/bash

# Base directory X
BASE_DIR="/Users/robertjin/Documents/GitHub/Trespass/model/vggface2_preprocessed/train/"

# Iterate through each subdirectory in BASE_DIR
totalfilecount=0
for dir in "$BASE_DIR"/*/; do
  if [ -d "$dir" ]; then
    # Count the number of regular files in the directory
    file_count=$(ls -l "$dir" | egrep -c '^-')
    totalfilecount=$((totalfilecount + file_count))
    # Print the directory name and the file count
  fi
done

echo "Directory: $dir has $totalfilecount files"