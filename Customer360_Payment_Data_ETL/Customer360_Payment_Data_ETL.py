import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import lit, when
from awsglue.dynamicframe import DynamicFrame
import time
import boto3
from botocore.exceptions import ClientError
import logging
from datetime import datetime
import json
import concurrent.futures
from pyspark.sql.functions import current_timestamp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timer function
def log_time_taken(start_time, operation_name):
    """
    Log the time taken for a specific operation.
    """
    end_time = time.time()
    logger.info(f"{operation_name} took {end_time - start_time:.2f} seconds.")

# Initialize Glue context and job
def initialize_glue_job():
    """
    Initialize the Glue job and its context.
    """
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args['JOB_NAME'], args)
    return glue_context, job

# Load Redshift credentials from AWS Secrets Manager
def get_secret():
    """
    Retrieve Redshift credentials from AWS Secrets Manager.
    """
    try:
        logger.info("Loading secrets from Secrets Manager...")
        
        # Resolve job parameters
        args = getResolvedOptions(sys.argv, ['SECRET_NAME'])
        secret_name = args['SECRET_NAME']
        
        args = getResolvedOptions(sys.argv, ['REGION_NAME'])
        region_name = args['REGION_NAME']
        
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region_name)

        # Fetch secret value
        response = client.get_secret_value(SecretId=secret_name)
        secret = response['SecretString'] if 'SecretString' in response else \
                 base64.b64decode(response['SecretBinary']).decode('utf-8')
                 
        return json.loads(secret)
    except ClientError as e:
        logger.error(f"Error retrieving secrets: {e}")
        raise

# Load data from S3
def load_data_from_s3(glue_context, s3_source_path):
    """
    Load data from S3 as a DynamicFrame.
    """
    try:
        logger.info("Loading data from S3...")
        s3_data_dyf = glue_context.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={
                "paths": [s3_source_path],
                "recurse": True,
                "jobBookmarkKeys": ["filename"], 
                "jobBookmarkKeysSortOrder": "asc"
            },
            transformation_ctx=f"Redshift_jb_payment_details",
            format="csv",
            format_options={"withHeader": True, "separator": ",", "quoteChar": "\""}
        )
        return s3_data_dyf
    except Exception as e:
        logger.error(f"Error loading data from S3: {e}")
        raise

# Transform the S3 data
def transform_s3_data(s3_data_dyf, glue_context):
    """
    Transform and map the S3 data DynamicFrame.
    """
    try:
        logger.info("Transforming S3 data...")
        transformed_dyf = s3_data_dyf.rename_field("customerID", "customer_id") \
            .rename_field("MonthlyCharges", "monthly_charges") \
            .rename_field("TotalCharges", "total_charges") \
            .rename_field("PaperlessBilling", "paperless_billing") \
            .apply_mapping([
                ("customer_id", "string", "customer_id", "string"),
                ("paperless_billing", "string", "paperless_billing", "boolean"),
                ("monthly_charges", "string", "monthly_charges", "decimal(10,2)"),
                ("total_charges", "string", "total_charges", "decimal(10,2)"),
                ("PaymentMethod", "string", "PaymentMethod", "string")
            ])
        df = transformed_dyf.toDF()
        df = df.withColumn("updated_on", current_timestamp())
        transformed_dyf = DynamicFrame.fromDF(df, glue_context, "joined_dyf_with_updated_on")
        return transformed_dyf
    except Exception as e:
        logger.error(f"Error transforming S3 data: {e}")
        raise

# Load payment method table from Redshift
def load_payment_method_table(glue_context, connection_options, table_name):
    """
    Load the payment method table from Redshift.
    """
    try:
        logger.info("Loading payment method data from Redshift...")
        
        # Add Snappy compression options
        compression_options = {
            "compression": "snappy"
        }
        # Merge compression options with existing Redshift options
        merged_connection_options = {
            **connection_options,  # Unpack original Redshift options
            **compression_options  # Add Snappy compression
        }
        
        payment_method_dyf = glue_context.create_dynamic_frame.from_options(
            connection_type="redshift",
            connection_options={
                "url": merged_connection_options["url"],
                "dbtable": table_name,
                "user": merged_connection_options["user"],
                "password": merged_connection_options["password"],
                "redshiftTmpDir": merged_connection_options["redshiftTmpDir"]
            }
        )
        return payment_method_dyf
    except Exception as e:
        logger.error(f"Error loading payment method table: {e}")
        raise

# Join S3 data with payment method data
def join_with_payment_method_id(s3_data_dyf, payment_method_dyf):
    """
    Perform join operation with payment method table.
    """
    try:
        logger.info("Joining S3 data with payment method data...")
        joined_data_dyf = s3_data_dyf.join(
            paths1=["PaymentMethod"], paths2=["payment_method"], frame2=payment_method_dyf
        )
        
        joined_data_dyf = joined_data_dyf.map(lambda record: {
            **record,
            "defaulter": False
        }).drop_fields(["payment_method", "PaymentMethod"])
        
        return joined_data_dyf
    except Exception as e:
        logger.error(f"Error during join operation: {e}")
        raise

# Write data to Redshift
def write_to_redshift(glue_context, dyf, connection_options, table_name):
    """
    Write the processed data to Redshift.
    """
    try:
        logger.info(f"Writing data to Redshift table: payment_details...")
        options = {
            **connection_options,
            "dbtable": f"public.{table_name}"
        }
        
        # Perform the write operation
        glue_context.write_dynamic_frame.from_options(
            frame=dyf,
            connection_type="redshift",
            connection_options=options,
            transformation_ctx="Redshift_jb_payment_details"
        )

    except Exception as e:
        logger.error(f"Error writing data to Redshift: {e}")
        raise

# Main ETL job execution with time logging
def main():
    """
    Main function to orchestrate the ETL process with time logging.
    """
    try:
        start_time = time.time()

        # Initialize job
        initialize_start = time.time()
        glue_context, job = initialize_glue_job()
        log_time_taken(initialize_start, "Job Initialization")

        # Global configurations -- Resolve job parameters
        args = getResolvedOptions(sys.argv, ['S3_SOURCE_PATH'])
        s3_source_path = args['S3_SOURCE_PATH']
        
        args = getResolvedOptions(sys.argv, ['REDSHIFT_TEMP_DIR'])
        redshift_tmp_dir = args['REDSHIFT_TEMP_DIR']

        payment_method_table = "public.dim_payment_method"

        # Fetch Redshift credentials
        secret_start = time.time()
        redshift_connection_options = get_secret()
        log_time_taken(secret_start, "Fetching Redshift Secrets")
            
        # Step 1: Read from S3 
        read_from_s3_start = time.time()
        s3_data_dyf = load_data_from_s3(glue_context, s3_source_path)
        log_time_taken(read_from_s3_start, "Reading S3 Data")
        
        # Step 2: Read from S3 
        read_from_redshift_start = time.time()
        payment_method_dyf = load_payment_method_table(glue_context, redshift_connection_options, payment_method_table)
        log_time_taken(read_from_redshift_start, "Reading Redshift Data")

        # Step 3: Transform S3 data
        transform_start = time.time()
        s3_data_dyf = transform_s3_data(s3_data_dyf, glue_context)
        log_time_taken(transform_start, "Transforming S3 Data")

        # Step 4: Join S3 data with payment method
        join_start = time.time()
        joined_dyf = join_with_payment_method_id(s3_data_dyf, payment_method_dyf)
        log_time_taken(join_start, "Joining S3 Data with Payment Method Data")

        # Step 5: Write data to Redshift
        write_start = time.time()
        write_to_redshift(glue_context, joined_dyf, redshift_connection_options, "payment_details")
        log_time_taken(write_start, "Writing Data to Redshift")

        log_time_taken(start_time, "Total ETL Job Execution Time")
        logger.info("ETL job completed successfully.")

    except Exception as e:
        logger.error(f"ETL job failed: {e}")

    finally:
        # Finalize job
        if 'job' in locals():
            job.commit()

if __name__ == "__main__":
    main()