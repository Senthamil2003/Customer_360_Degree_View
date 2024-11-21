import logging
import sys
import traceback
import boto3
import json
import base64
from pyspark.sql import functions as F
from pyspark.sql.functions import col, trim, lower, current_timestamp, lit, regexp_replace, split
from pyspark.sql.types import BooleanType, IntegerType
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()



# -----------------------------------
# Initialize AWS Glue job arguments and context
# -----------------------------------

try:
    # Log the initialization of Glue job arguments
    logger.info("Initializing AWS Glue job arguments and context...")

    # Parse job arguments
    ARGS = getResolvedOptions(sys.argv, ['JOB_NAME'])

    # Initialize SparkContext
    SC = SparkContext()
    logger.info("SparkContext initialized.")

    # Create a GlueContext
    GLUE_CONTEXT = GlueContext(SC)
    logger.info("GlueContext created.")

    # Access SparkSession
    SPARK = GLUE_CONTEXT.spark_session
    logger.info("SparkSession accessed.")

    # Initialize Glue Job
    JOB = Job(GLUE_CONTEXT)
    JOB.init(ARGS['JOB_NAME'], ARGS)
    logger.info(f"Glue job '{ARGS['JOB_NAME']}' initialized successfully.")

except Exception as e:
    # Log any errors that occur during initialization
    logger.critical("Failed to initialize AWS Glue job arguments and context.")
    logger.critical(f"Error details: {e}")
    raise e

# Retrieve job parameters
args = getResolvedOptions(
    sys.argv, 
    ['s3_source_path', 'redshift_tmp_dir', 'secret_name', 'region_name']
)

# Assign parameters to variables
S3_SOURCE_PATH = args['s3_source_path']
REDSHIFT_TMP_DIR = args['redshift_tmp_dir']
SECRET_NAME = args['secret_name']
REGION_NAME = args['region_name']

# Print values to verify
print(f"S3 Source Path: {S3_SOURCE_PATH}")
print(f"Redshift Temp Directory: {REDSHIFT_TMP_DIR}")
print(f"Secret Name: {SECRET_NAME}")
print(f"Region Name: {REGION_NAME}")


# # Global configuration
# S3_SOURCE_PATH = "s3://customer360-v1-s3bucket/raw/year=2024/"
# REDSHIFT_TMP_DIR = "s3://aws-glue-assets-894811220469-us-east-1/temporary/"
# SECRET_NAME = "customer360_redshift_secrets"
# REGION_NAME = "us-east-1"

# -----------------------------------
# Function to fetch secrets from AWS Secrets Manager
# -----------------------------------
def get_secret():
    """
    Fetch secrets from AWS Secrets Manager.
    """
    logger.info("Attempting to load secrets from AWS Secrets Manager...")
    try:
        # Create a boto3 session and Secrets Manager client
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=REGION_NAME)

        # Fetch the secret value
        get_secret_value_response = client.get_secret_value(SecretId=SECRET_NAME)
        secret = get_secret_value_response.get('SecretString') or \
                 base64.b64decode(get_secret_value_response['SecretBinary']).decode('utf-8')

        logger.info("Secrets successfully fetched from Secrets Manager.")
        return json.loads(secret)

    except ClientError as e:
        # Handle AWS client-specific errors
        logger.error(f"Error fetching secrets from Secrets Manager: {e}")
        raise e
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error while fetching secrets: {e}")
        raise e

# -----------------------------------
# Fetch Redshift connection options
# -----------------------------------
try:
    logger.info("Fetching Redshift connection options from Secrets Manager...")
    SECRET_DATA = get_secret()

    # Populate Redshift connection options
    REDSHIFT_CONNECTION_OPTIONS = {
        "url": SECRET_DATA["url"],
        "user": SECRET_DATA["user"],
        "password": SECRET_DATA["password"],
        "redshiftTmpDir": REDSHIFT_TMP_DIR
    }
    logger.info("Redshift connection options successfully fetched.")

except Exception as e:
    # Log critical failure and exit
    logger.critical(f"Failed to fetch Redshift connection options: {e}")
    logger.critical("Exiting due to configuration error.")
    sys.exit(1)


# -----------------------------------
# Function to load data from S3
# -----------------------------------
def load_data_from_s3():
    """
    Load raw data from S3 as a Glue DynamicFrame.

    Returns:
        DynamicFrame: The loaded data as a Glue DynamicFrame.
    """
    try:
        logger.info("Starting to load data from S3...")

        # Load data from S3 into a Glue DynamicFrame
        input_data = GLUE_CONTEXT.create_dynamic_frame.from_options(
            format_options={"quoteChar": "\"", "withHeader": True, "separator": ","},
            connection_type="s3",
            format="csv",
            connection_options={
                "paths": [S3_SOURCE_PATH],
                "recurse": True,
                # "groupFiles": "inPartition",
                # "jobBookmarkKeys": ["filename"],  # Use filename as the bookmark key
                # "jobBookmarkKeysSortOrder": "asc"
            },
            transformation_ctx="Customer360_Customer_Info_Details"
        )

        # Log the record count
        record_count = input_data.count()
        logger.info(f"Data successfully loaded from S3. Total records: {record_count}")

        return input_data

    except Exception as e:
        # Log the error and re-raise the exception
        logger.error(f"Failed to load data from S3: {e}")
        logger.error(traceback.format_exc())
        raise e

# -----------------------------------
# Function to transform a specific column in a DynamicFrame
# -----------------------------------

from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import current_timestamp, col

from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import current_timestamp, col, lower, trim, lit

def transform_column_dyf(dyf, column_name, new_column_name):
    """
    Apply transformations to a specific column in the DynamicFrame.

    Args:
        dyf (DynamicFrame): Input DynamicFrame.
        column_name (str): The name of the column to transform.
        new_column_name (str): The name of the new column after transformation.

    Returns:
        DynamicFrame: A transformed DynamicFrame with the new column.

    Raises:
        RuntimeError: If the column does not exist or any other error occurs.
    """
    try:
        logger.info(f"Starting transformation on column '{column_name}'...")

        # Validate if the column exists in the DynamicFrame
        columns = [field.name for field in dyf.schema()]
        if column_name not in columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DynamicFrame. "
                             f"Available columns: {columns}")

        # Convert DynamicFrame to DataFrame for transformations
        logger.info("Converting DynamicFrame to DataFrame for transformations...")
        df = dyf.toDF()

        # Step 1: Drop duplicates
        logger.info("Dropping duplicates...")
        df = df.dropDuplicates([column_name])

        # Step 2: Rename the column and apply transformations
        logger.info(f"Renaming column '{column_name}' to '{new_column_name}' and applying transformations...")
        transformed_df = df.withColumnRenamed(column_name, new_column_name) \
            .withColumn(new_column_name, lower(trim(col(new_column_name)))) \
            .filter(col(new_column_name).isNotNull()) \
            .withColumn("updated_on", lit(current_timestamp()))  # Add current timestamp

        # Step 3: Convert back to DynamicFrame
        logger.info("Converting back to DynamicFrame...")
        transformed_dyf = DynamicFrame.fromDF(transformed_df, GLUE_CONTEXT, "transformed_dynamic_frame")

        # Step 4: Select specific column and timestamp
        logger.info(f"Selecting columns: '{new_column_name}' and 'updated_on'...")
        transformed_dyf = transformed_dyf.select_fields([new_column_name, "updated_on"])

        # Log success and schema for debugging
        logger.info(f"Column '{column_name}' successfully transformed into '{new_column_name}'.")
        logger.info(f"Schema of transformed DynamicFrame: {[field.name for field in transformed_dyf.schema()]}")

        return transformed_dyf

    except ValueError as ve:
        # Log specific column validation errors
        logger.error(f"Validation error: {ve}")
        raise ve
    except Exception as e:
        # Log and re-raise unexpected errors
        logger.error(f"Error transforming column '{column_name}' to '{new_column_name}': {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to transform column '{column_name}' to '{new_column_name}': {e}")

# -----------------------------------
# Function to load existing data from Redshift as a DynamicFrame
# -----------------------------------
def load_existing_data_from_redshift_dyf(table_name, column_name):
    """
    Load existing data from a specified Redshift table as a DynamicFrame for comparison.

    Args:
        table_name (str): The name of the Redshift table.
        column_name (str): The column to select from the Redshift table.

    Returns:
        DynamicFrame: A DynamicFrame containing the selected column data.

    Raises:
        RuntimeError: If there is an error while fetching data from Redshift.
    """
    try:
        logger.info(f"Loading existing data from Redshift table '{table_name}', column '{column_name}'...")

        # Load data from Redshift into a DynamicFrame
        existing_data_dyf = GLUE_CONTEXT.create_dynamic_frame.from_options(
            connection_type="redshift",
            connection_options={
                "url": REDSHIFT_CONNECTION_OPTIONS["url"],
                "user": REDSHIFT_CONNECTION_OPTIONS["user"],
                "password": REDSHIFT_CONNECTION_OPTIONS["password"],
                "dbtable": table_name,
                "redshiftTmpDir": REDSHIFT_TMP_DIR  # Ensure this points to a valid S3 path
            }
        )

        # Validate column existence in the DynamicFrame
        existing_columns = [field.name for field in existing_data_dyf.schema()]
        if column_name not in existing_columns:
            raise ValueError(f"Column '{column_name}' not found in Redshift table '{table_name}'. Available columns: {existing_columns}")

        # Select the specific column
        existing_data_dyf = existing_data_dyf.select_fields([column_name])

        # Check if the DynamicFrame is empty
        record_count = existing_data_dyf.count()
        if record_count == 0:
            logger.warning(f"No records found in Redshift table '{table_name}' for column '{column_name}'. Returning an empty DynamicFrame.")
            return existing_data_dyf

        # Log success message
        logger.info(f"Successfully loaded {record_count} records from Redshift table '{table_name}', column '{column_name}'.")

        return existing_data_dyf

    except Exception as e:
        # Log the error and re-raise as RuntimeError
        logger.error(f"Error loading data from Redshift table '{table_name}': {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Error loading existing data from {table_name}: {e}")


# -----------------------------------
# Function to filter new records
# -----------------------------------
def filter_new_records_dyf(transformed_dyf, existing_dyf, column_name):
    """
    Filter out records from the transformed DynamicFrame that are already present in the existing DynamicFrame.

    Args:
        transformed_dyf (DynamicFrame): The DynamicFrame containing new records to be filtered.
        existing_dyf (DynamicFrame): The DynamicFrame containing existing records for comparison.
        column_name (str): The column name used for comparison.

    Returns:
        DynamicFrame: A DynamicFrame containing only the new records not present in the existing DynamicFrame.

    Raises:
        ValueError: If the column name is not found in the transformed DynamicFrame.
        RuntimeError: If an unexpected error occurs during filtering.
    """
    try:
        logger.info(f"Filtering new records based on column '{column_name}'...")

        # Log record count and schema of transformed DynamicFrame
        transformed_count = transformed_dyf.count()
        logger.info(f"Record count in transformed DynamicFrame: {transformed_count}")
        logger.info(f"Schema of transformed DynamicFrame: {[field.name for field in transformed_dyf.schema()]}")

        # Validate column existence in transformed DynamicFrame
        transformed_columns = [field.name for field in transformed_dyf.schema()]
        if column_name not in transformed_columns:
            raise ValueError(f"Column '{column_name}' not found in transformed DynamicFrame. Available columns: {transformed_columns}")

        # Log record count and schema of existing DynamicFrame
        existing_count = existing_dyf.count()
        logger.info(f"Record count in existing DynamicFrame: {existing_count}")
        logger.info(f"Schema of existing DynamicFrame: {[field.name for field in existing_dyf.schema()]}")

        # Handle case where existing DynamicFrame is empty
        if existing_count == 0:
            logger.warning(f"Existing DynamicFrame is empty. All records from transformed DynamicFrame will be treated as new.")
            return transformed_dyf

        # Validate column existence in existing DynamicFrame
        existing_columns = [field.name for field in existing_dyf.schema()]
        if column_name not in existing_columns:
            raise ValueError(f"Column '{column_name}' not found in existing DynamicFrame. Available columns: {existing_columns}")

        # Perform the join operation
        logger.info("Performing join operation...")
        joined_dyf = transformed_dyf.join(
            paths1=[column_name],
            paths2=[column_name],
            frame2=existing_dyf,
            transformation_ctx="joined_dyf"
        )

        # Log record count after join
        joined_count = joined_dyf.count()
        logger.info(f"Record count in joined DynamicFrame: {joined_count}")

        # Filter out records that matched (exclude records where column from existing_dyf is present)
        logger.info("Filtering unmatched records...")
        def filter_non_matching_records(rec):
            # Check if the matching column from the second frame is None
            return rec[f"{column_name}_right"] is None

        new_records_dyf = joined_dyf.filter(filter_non_matching_records)

        # Log the success message with the count of new records
        new_records_count = new_records_dyf.count()
        logger.info(f"Successfully filtered new records. Count of new records: {new_records_count}")

        return new_records_dyf

    except ValueError as ve:
        # Log and raise specific column-related errors
        logger.error(f"Column validation error: {ve}")
        raise ve

    except Exception as e:
        # Log and raise general filtering errors
        logger.error(f"Error filtering new records based on column '{column_name}': {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Error filtering new records: {e}")



# -----------------------------------
# Function to insert new records into Redshift
# -----------------------------------
def insert_new_records_to_redshift_dyf(new_records_dyf, table_name, transformation_ctx):
    """
    Insert new records into a specified Redshift table if there are records to insert.

    Args:
        new_records_dyf (DynamicFrame): The DynamicFrame containing new records to be inserted.
        table_name (str): The name of the Redshift table where the records will be inserted.
        transformation_ctx (str): The transformation context for Glue.

    Raises:
        RuntimeError: If an error occurs during the insertion process.
    """
    try:
        # Check if there are records to insert
        record_count = new_records_dyf.count()
        if record_count > 0:
            logger.info(f"Inserting {record_count} new records into Redshift table '{table_name}'...")

            # Write the new records to Redshift
            GLUE_CONTEXT.write_dynamic_frame.from_options(
                frame=new_records_dyf,
                connection_type="redshift",
                connection_options={
                    **REDSHIFT_CONNECTION_OPTIONS,
                    "dbtable": table_name,
                },
                transformation_ctx=transformation_ctx
            )

            logger.info(f"Successfully inserted {record_count} new records into '{table_name}'.")

        else:
            # Log when there are no records to insert
            logger.info(f"No new records to insert into Redshift table '{table_name}'.")

    except Exception as e:
        # Log and raise any errors during the insertion process
        logger.error(f"Error inserting records into Redshift table '{table_name}': {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to insert records into '{table_name}': {e}")



# -----------------------------------
# Function to handle loading data into a dimension table
# -----------------------------------
def load_dim_table_dyf(dyf, column_name, new_column_name, table_name, transformation_ctx):
    """
    Generic function to handle loading data into a dimension table in Redshift using DynamicFrame.

    Args:
        dyf (DynamicFrame): The input DynamicFrame containing source data.
        column_name (str): The column name to be transformed from the source data.
        new_column_name (str): The new column name after transformation.
        table_name (str): The name of the Redshift dimension table.
        transformation_ctx (str): The transformation context for Glue.

    Raises:
        RuntimeError: If any step in the loading process fails.
    """
    try:
        logger.info(f"Starting to load data into dimension table '{table_name}'...")

        # Transform the specified column
        logger.info(f"Transforming column '{column_name}' to '{new_column_name}'...")
        transformed_dyf = transform_column_dyf(dyf, column_name, new_column_name)
        logger.info(f"Column '{column_name}' successfully transformed.")

        # Load existing data from Redshift
        logger.info(f"Loading existing data from Redshift table '{table_name}' for comparison...")
        existing_dyf = load_existing_data_from_redshift_dyf(table_name, new_column_name)
        logger.info(f"Existing data successfully loaded from Redshift table '{table_name}'.")

        # Filter out existing records
        logger.info(f"Testing...................")
        logger.info(f"Schema of existing DynamicFrame: {[field.name for field in existing_dyf.schema()]}")
        logger.info(f"Schema of transformed DynamicFrame: {[field.name for field in transformed_dyf.schema()]}")
        logger.info(f"Filtering new records for table '{table_name}'...")
        
        new_records_dyf = filter_new_records_dyf(transformed_dyf, existing_dyf, new_column_name)
        logger.info(f"Filtered new records. Count of new records: {new_records_dyf.count()}")

        # Insert new records into Redshift
        logger.info(f"Inserting new records into dimension table '{table_name}'...")
        insert_new_records_to_redshift_dyf(new_records_dyf, table_name, transformation_ctx)
        logger.info(f"Data successfully loaded into dimension table '{table_name}'.")

    except Exception as e:
        # Log and raise any errors encountered during the process
        logger.error(f"Error loading data into dimension table '{table_name}': {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load data into dimension table '{table_name}': {e}")
        
def load_dim_tables_parallel(s3_dynamic_frame):
    """
    Load dimension tables in parallel using ThreadPoolExecutor.

    Args:
        s3_dynamic_frame (DynamicFrame): The input DynamicFrame containing source data.

    Raises:
        RuntimeError: If an error occurs while loading dimension tables in parallel.
    """
    try:
        logger.info("Starting parallel dimension table loading...")

        # Define the dimension table loading functions with their specific parameters
        dim_table_tasks = [
            {
                'func': load_dim_table_dyf,
                'args': {
                    'dyf': s3_dynamic_frame,
                    'column_name': 'Location',
                    'new_column_name': 'locationname',
                    'table_name': 'public.dim_location_details',
                    'transformation_ctx': 'Redshift_Location_Upsert'
                }
            },
            {
                'func': load_dim_table_dyf,
                'args': {
                    'dyf': s3_dynamic_frame,
                    'column_name': 'Contract',
                    'new_column_name': 'contractname',
                    'table_name': 'public.dim_contract_details',
                    'transformation_ctx': 'Redshift_Contract_Upsert'
                }
            }
        ]

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=len(dim_table_tasks)) as executor:
            # Submit tasks to the executor
            futures = {
                executor.submit(task['func'], **task['args']): task 
                for task in dim_table_tasks
            }

            # Wait for all tasks to complete and handle results/exceptions
            for future in as_completed(futures):
                task = futures[future]
                try:
                    # Check if the task raised an exception
                    future.result()
                    logger.info(f"Successfully completed loading task for {task['args']['table_name']}")
                except Exception as e:
                    logger.error(f"Error loading dimension table {task['args']['table_name']}: {e}")
                    raise  # Re-raise the exception to be caught by the main error handler

        logger.info("Parallel dimension table loading completed successfully.")

    except Exception as e:
        # Log and raise any errors that occur during parallel processing
        logger.error(f"Error in parallel dimension table loading: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load dimension tables in parallel: {e}")

# Main ETL job execution
try:
    logger.info("Starting ETL job...")

    # Step 1: Load raw data from S3
    logger.info("Loading raw data from S3...")
    s3_dynamic_frame = load_data_from_s3()
    logger.info(f"Total records loaded from S3: {s3_dynamic_frame.count()}")

    # Step 2: Load and process dimension tables in parallel
    logger.info("Loading and processing dimension tables in parallel...")
    load_dim_tables_parallel(s3_dynamic_frame)
    logger.info("Dimension tables loaded and processed successfully.")

    logger.info("ETL job completed successfully.")

except Exception as e:
    # Log any errors that occur during the ETL process
    logger.error(f"ETL job failed: {e}")
    logger.error(traceback.format_exc())

finally:
    # Commit the Glue job
    try:
        JOB.commit()
        logger.info("Job committed successfully.")
    except Exception as commit_error:
        logger.error(f"Failed to commit the job: {commit_error}")
        logger.error(traceback.format_exc())