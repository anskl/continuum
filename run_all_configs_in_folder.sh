#!/bin/bash
CONFIGURATION_FOLDER="configuration/experiment_kata"

for file in "$CONFIGURATION_FOLDER"/*.cfg; do

    config_file=$(basename -- "$file")

    output_filename="$CONFIGURATION_FOLDER/${config_file%.*}_no_skew_20s.txt"

    # echo "$output_filename"
    echo "$(date +"%T") python3 continuum.py \"$file\" > \"$output_filename\""
    python3 continuum.py "$file" > "$output_filename"
done
