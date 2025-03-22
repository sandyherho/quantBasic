#!/bin/bash

# Simple script to automatically add, commit, and push changes to GitHub
# Usage: ./gitpush.sh "Optional commit message"

# Set default commit message if none provided
if [ -z "$1" ]; then
  COMMIT_MESSAGE="Update $(date +"%Y-%m-%d %H:%M:%S")"
  echo "No commit message provided. Using default: \"$COMMIT_MESSAGE\""
else
  COMMIT_MESSAGE="$1"
fi

# Add all changes
git add .

# Commit with the message
git commit -m "$COMMIT_MESSAGE"

# Push to the main branch
git push origin main

echo "Changes successfully pushed to origin/main"
