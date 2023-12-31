#!/bin/bash

# Parent directory
dir="/Users/saip/My Drive/machine-learning-fundamentals/deep_learning"

# List of subdirectories to clear
subdirs=("tensorboard_logs" "logs")

# Loop over subdirectories
for subdir in "${subdirs[@]}"; do
  # Construct the full path
  full_path="$dir/$subdir"

  # Check if the directory exists
  if [[ -d "$full_path" ]]; then
    # Remove all files in the directory
    rm -r "$full_path"/*
    echo "All files in $full_path have been removed."
  else
    echo "Directory $full_path does not exist."
  fi
done
