import sys
import time
import logging
import boto3
from botocore.exceptions import ClientError
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import os
import json
import concurrent.futures
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import broadcast

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



# Check if the script is running locally by verifying the presence of the JSON config file
CONFIG_FILE_PATH = 'Customer360_payment_method_etl_dynamicFrame.json'
if os.path.exists(CONFIG_FILE_PATH):
    logger.info(f"Running locally. Loading configuration from {CONFIG_FILE_PATH}...")

    # Load parameters from JSON configuration file
    with open(CONFIG_FILE_PATH, 'r') as config_file:
        config = json.load(config_file)

    # Extract arguments from JSON
    args = {
        'JOB_NAME': config['name'],
        's3_source_path': config['defaultArguments']['--s3_source_path'],
        'redshift_tmp_dir': config['defaultArguments']['--redshift_tmp_dir'],
        'secret_name': config['defaultArguments']['--secret_name'],
        'region_name': config['defaultArguments']['--region_name']
    }
else:
    logger.info("Running in AWS Glue environment. Parsing command line arguments...")
    # Parse command line job parameters (AWS Glue environment)
    args = getResolvedOptions(
        sys.argv,
        ['JOB_NAME', 's3_source_path', 'redshift_tmp_dir', 'secret_name', 'region_name']
    )

# Initialize Spark, GlueContext, and Job
sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args['JOB_NAME'], args)

# Set up variables from parameters
S3_SOURCE_PATH = args['s3_source_path']
REDSHIFT_TMP_DIR = args['redshift_tmp_dir']
PAYMENT_METHOD_TABLE = "public.dim_payment_method"
SECRET_NAME = args['secret_name']
REGION_NAME = args['region_name']

# Decorator to log the execution time of functions
def log_time(func):
    """Decorator to measure and log the execution time of functions."""
    def wrapper(*args, **kwargs):
        logger.info(f"Starting function: {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"An error occurred in function {func.__name__}: {e}")
            raise e  # Re-raise to propagate the exception
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"Function {func.__name__} completed in {elapsed_time:.2f} seconds.")
    return wrapper

# Load secret for Redshift connection
@log_time
def get_secret():
    logger.info("Loading secrets from Secrets Manager...")
    secret_name = SECRET_NAME
    region_name = REGION_NAME

    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response.get('SecretString')
        return json.loads(secret)

    except ClientError as e:
        logger.error(f"An error occurred while loading secrets: {e}")
        raise e

SECRET_DATA = get_secret()
redshift_connection_options = {
    "url": "jdbc:redshift://localhost:5439/customer360_db_redshift",
    # "url": SECRET_DATA["url"],
    "dbtable": PAYMENT_METHOD_TABLE,
    "user": SECRET_DATA["user"],
    "password": SECRET_DATA["password"],
    "redshiftTmpDir": REDSHIFT_TMP_DIR
}

# Load data from S3
@log_time
def load_data_from_s3():
    logger.info("Loading data from S3...")

    try:
        s3_data_dyf = glue_context.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={
                "paths": [S3_SOURCE_PATH],
                "recurse": True,
                "groupFiles": "inPartition",
                "jobBookmarkKeys": ["filename"],
                "jobBookmarkKeysSortOrder": "asc"
            },
            transformation_ctx="S3_to_Payment_Method_Details",
            format="csv",
            format_options={"withHeader": True, "separator": ",", "quoteChar": "\""}
        )

        logger.info("Data successfully loaded from S3.")
        return s3_data_dyf

    except Exception as e:
        logger.error(f"Failed to load data from S3: {e}")
        raise e

# Load existing data from Redshift
@log_time
def load_existing_data():
    logger.info(f"Loading existing data from Redshift table: {PAYMENT_METHOD_TABLE} for incremental load...")

    try:
        existing_data_dyf = glue_context.create_dynamic_frame.from_options(
            connection_type="redshift",
            connection_options=redshift_connection_options
        )

        logger.info(f"Data successfully loaded from Redshift table: {PAYMENT_METHOD_TABLE}")
        return existing_data_dyf

    except Exception as e:
        logger.error(f"Failed to load data from Redshift table {PAYMENT_METHOD_TABLE}: {e}")
        raise e

# Preprocess the S3 data for loading into Redshift
@log_time
def preprocess_payment_method_data(s3_data_dyf, glue_context):
    logger.info("Preprocessing PaymentMethod data...")

    try:
        # Rename column and filter invalid rows
        renamed_dyf = s3_data_dyf.rename_field("PaymentMethod", "payment_method")
        filtered_dyf = renamed_dyf.filter(
            lambda row: row["payment_method"] is not None and row["payment_method"].strip() != ""
        )

        # Remove duplicates
        filtered_df = filtered_dyf.toDF()
        distinct_df = filtered_df.select("payment_method").distinct()

        # Convert back to DynamicFrame
        distinct_dyf = DynamicFrame.fromDF(distinct_df, glue_context, "distinct_payment_method")

        logger.info(f"Preprocessed PaymentMethod values count: {distinct_dyf.count()}")
        return distinct_dyf

    except Exception as e:
        logger.error(f"Error preprocessing PaymentMethod data: {e}")
        raise e

# Perform incremental load transformation
@log_time
def filter_incremental_data(new_data_dyf, existing_data_dyf):
    logger.info("Performing incremental load transformation with deduplication...")

    try:
        new_data_df = new_data_dyf.toDF()
        existing_data_df = existing_data_dyf.toDF()

        # Deduplicate new data
        new_data_df = new_data_df.dropDuplicates(['payment_method'])

        # Get incremental records by comparing against existing data
        incremental_data_df = new_data_df.join(
            broadcast(existing_data_df),
            on="payment_method",
            how="left_anti"
        )

        # Convert back to DynamicFrame
        incremental_data_dyf = DynamicFrame.fromDF(incremental_data_df, glue_context, "incremental_data")

        logger.info("Incremental transformation completed successfully.")
        return incremental_data_dyf

    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        raise e

# Write data to Redshift
@log_time
def write_to_redshift(dynamic_frame):
    logger.info(f"Checking if there are records to write to Redshift table: {PAYMENT_METHOD_TABLE}...")

    record_count = dynamic_frame.count()  # Get the count of records in the DynamicFrame

    if record_count > 0:
        logger.info(f"Records found ({record_count}), writing data to Redshift table: {PAYMENT_METHOD_TABLE}...")

        try:
            glue_context.write_dynamic_frame.from_options(
                frame=dynamic_frame,
                connection_type="redshift",
                connection_options=redshift_connection_options,
                transformation_ctx=f"Redshift_job_bookmark_{PAYMENT_METHOD_TABLE}"
            )

            logger.info(f"Data successfully written to Redshift table: {PAYMENT_METHOD_TABLE}")

        except Exception as e:
            logger.error(f"Failed to write data to Redshift table {PAYMENT_METHOD_TABLE}: {e}")
            raise e
    else:
        logger.info(f"No records to write to Redshift table: {PAYMENT_METHOD_TABLE}")


# Perform load operation to Redshift
@log_time
def perform_load(preprocessed_data, existing_data=None):
    try:
        if existing_data is not None and existing_data.count() > 0:
            logger.info("Performing incremental load...")
            transformed_data = filter_incremental_data(preprocessed_data, existing_data)
        else:
            logger.info("No existing data found. Performing full load...")
            transformed_data = preprocessed_data

        # Attempt to write data to Redshift
        write_to_redshift(transformed_data)

    except Exception as e:
        logger.error(f"An error occurred during the load operation: {e}")
        raise e  # Re-raise the exception to handle it at a higher level if needed

# Main function to orchestrate the ETL process
@log_time
def main():
    try:
        logger.info("ETL job started.")

        # Load S3 data and Redshift data concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_s3_payment_method_data = executor.submit(load_data_from_s3)
            future_existing_payment_method_data = executor.submit(load_existing_data)

            # Preprocess the data as soon as it's loaded from S3
            logger.debug("Waiting for S3 data to be loaded...")
            s3_payment_method_data = future_s3_payment_method_data.result()
            logger.debug("S3 data loaded successfully.")

            logger.debug("Preprocessing S3 data...")
            preprocessed_payment_method_data = preprocess_payment_method_data(s3_payment_method_data, glue_context)
            logger.debug("S3 data preprocessed successfully.")

            # Wait for existing data from Redshift
            logger.debug("Waiting for existing data to be loaded from Redshift...")
            existing_payment_method_data = future_existing_payment_method_data.result()
            logger.debug("Existing data loaded successfully.")

        # Perform the load operation to Redshift
        perform_load(preprocessed_payment_method_data, existing_payment_method_data)

        logger.info("ETL job completed successfully.")

    except Exception as e:
        logger.error(f"ETL job failed: {e}")
        raise e

if __name__ == "__main__":
    main()
    job.commit()
