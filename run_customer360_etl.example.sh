#!/bin/bash

# Define the path to the directory containing the Python script
PYTHON_SCRIPT_PATH="/path/to/your/workspace/folder_containing_python_script"
# Define the name of the Python script file
PYTHON_SCRIPT_FILE="python_script.py"

# Define configuration variables (update these with your actual values)
JOB_NAME="Your_Job_Name"
S3_SOURCE_PATH="s3://your-s3-source-bucket/raw/year=YYYY/"
REDSHIFT_TMP_DIR="s3://your-s3-temp-directory/"
SECRET_NAME="your_redshift_secret_name"
REGION_NAME="your-region"

# Execute the Python script
python3 "${PYTHON_SCRIPT_PATH}/${PYTHON_SCRIPT_FILE}" \
  --JOB_NAME "$JOB_NAME" \
  --s3_source_path "$S3_SOURCE_PATH" \
  --redshift_tmp_dir "$REDSHIFT_TMP_DIR" \
  --secret_name "$SECRET_NAME" \
  --region_name "$REGION_NAME"

# NOTE:
# 1. Replace `/path/to/your/workspace` with the actual path to your script directory.
# 2. Replace "Your_Job_Name", S3 paths, secret name, and region with actual values.
# 3. Save this file as `run_customer360_etl.sh` and customize it as needed.
