#!/bin/bash

# Define source and destination folders
SOURCE_FOLDER="dataset_artworks"
DEST_FOLDER="resized_dataset_artworks"
SIZE="56x56"

# Create the destination folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

# Loop through each image in the source folder
find "$SOURCE_FOLDER" -type f -name "*.jpg" | while read -r file; do
    # Get the relative path and directory structure of the current file
    RELATIVE_PATH="${file#$SOURCE_FOLDER/}"
    DEST_PATH="$DEST_FOLDER/$RELATIVE_PATH"
    
    # Create the destination directory structure if it doesn't exist
    mkdir -p "$(dirname "$DEST_PATH")"
    
    # Use ffmpeg to resize the image and save it to the new location
    ffmpeg -i "$file" -vf "scale=$SIZE" "$DEST_PATH"
done
