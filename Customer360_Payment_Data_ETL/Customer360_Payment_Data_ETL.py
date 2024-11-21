import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import lit, when, expr
import pyspark.sql.functions as F
from awsglue.dynamicframe import DynamicFrame
import time
import boto3
from botocore.exceptions import ClientError
import logging
import pytz
from datetime import datetime
import json
import concurrent.futures
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

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
def get_secret(region_name="us-east-1"):
    """
    Retrieve Redshift credentials from AWS Secrets Manager.
    """
    try:
        logger.info("Loading secrets from Secrets Manager...")
        
        # Resolve job parameters
        args = getResolvedOptions(sys.argv, ['SECRET_NAME'])
        secret_name = args['SECRET_NAME']
        
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
            transformation_ctx=f"Redshift_job_bookmark_payment_details",
            format="csv",
            format_options={"withHeader": True, "separator": ",", "quoteChar": "\""}
        )
        return s3_data_dyf
    except Exception as e:
        logger.error(f"Error loading data from S3: {e}")
        raise
    
# Fetch the last payment_id from Redshift
def get_last_payment_id_from_redshift(glue_context, connection_options, table_name):
    """
    Fetch the last payment_id from Redshift, and handle the case where the table does not exist.
    """
    try:
        logger.info(f"Fetching last payment_id from Redshift table: {table_name}...")
        
        # Create a Spark session for querying Redshift
        redshift_df = glue_context.read.format("jdbc").options(
            url=connection_options["url"],
            dbtable=f"public.{table_name}",
            user=connection_options["user"],
            password=connection_options["password"]
        ).load()
        
        # Get the max payment_id from the table
        last_payment_id = redshift_df.agg({"payment_id": "max"}).collect()[0][0]
        
        return last_payment_id if last_payment_id is not None else 0  # Return 0 if no payment_id exists
    except Exception as e:
        logger.error(f"Error fetching last payment_id from Redshift: {e}")
        
        # Handle table not found error or any other exceptions by setting last_payment_id to 0
        logger.info(f"Table not found or error occurred, defaulting last_payment_id to 0.")
        return 0

# Transform the S3 data
def transform_s3_data(s3_data_dyf, glue_context, last_payment_id):
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
                ("monthly_charges", "string", "monthly_charges", "double"),
                ("total_charges", "string", "total_charges", "double"),
                ("PaymentMethod", "string", "PaymentMethod", "string")
            ])
            
        ist_timezone = pytz.timezone("Asia/Kolkata")
        ist_time = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M:%S")
        transformed_dyf = transformed_dyf.map(lambda record: {
            **record,
            "defaulter": False,
            "updated_on": ist_time
        })
        
        # Convert DynamicFrame to DataFrame for transformation
        transformed_df = transformed_dyf.toDF()

        # Add a self-incrementing column `payment_id`
        window_spec = Window.orderBy(F.lit(1))  
        
        transformed_df = transformed_df.withColumn(
            "payment_id", F.row_number().over(window_spec) + last_payment_id
        )
        
        transformed_dyf = DynamicFrame.fromDF(transformed_df, glue_context, "transformed_data")
            
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
        payment_method_dyf = glue_context.create_dynamic_frame.from_options(
            connection_type="redshift",
            connection_options={
                "url": connection_options["url"],
                "dbtable": table_name,
                "user": connection_options["user"],
                "password": connection_options["password"],
                "redshiftTmpDir": connection_options["redshiftTmpDir"]
            }
        )
        return payment_method_dyf
    except Exception as e:
        logger.error(f"Error loading payment method table: {e}")
        raise

# Join S3 data with payment method data
def join_with_payment_method_id(s3_data_dyf, payment_method_dyf):
    """
    Perform join operation with payment method table and update timestamp.
    """
    try:
        logger.info("Joining S3 data with payment method data...")
        joined_data_dyf = s3_data_dyf.join(
            paths1=["PaymentMethod"], paths2=["payment_method"], frame2=payment_method_dyf
        )
        
        joined_data_dyf = joined_data_dyf.drop_fields(["payment_method", "PaymentMethod"])
        
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
        glue_context.write_dynamic_frame.from_options(
            frame=dyf,
            connection_type="redshift",
            connection_options={
                "url": connection_options["url"],
                "dbtable": f"public.{table_name}",
                "user": connection_options["user"],
                "password": connection_options["password"],
                "redshiftTmpDir": connection_options["redshiftTmpDir"],
                "connectionName": connection_options["connectionName"]
            },
            transformation_ctx=f"Redshift_job_bookmark_payment_details"
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

        # Global configurations
        s3_source_path = "s3://customer360-v1-s3bucket/raw/year=2024/"
        redshift_tmp_dir = "s3://aws-glue-assets-894811220469-us-east-1/temporary/"
        payment_method_table = "public.dim_payment_method"

        # Fetch Redshift credentials
        secret_start = time.time()
        redshift_connection_options = get_secret()
        log_time_taken(secret_start, "Fetching Redshift Secrets")
            
        # Step 1: Read from S3 
        read_from_s3_start = time.time()
        s3_data_dyf = load_data_from_s3(glue_context, s3_source_path)
        log_time_taken(read_from_s3_start, "Reading S3 Data")
        
        # Step 2: Load Payment Method Data from Redshift
        read_from_redshift_start = time.time()
        payment_method_dyf = load_payment_method_table(glue_context, redshift_connection_options, payment_method_table)
        log_time_taken(read_from_redshift_start, "Reading Redshift Data")
        
        # Step 3: Get last Id
        last_payment_id = get_last_payment_id_from_redshift(glue_context, redshift_connection_options, "payment_details")

        # Step 4: Transform S3 data
        transform_start = time.time()
        s3_data_dyf = transform_s3_data(s3_data_dyf, glue_context, last_payment_id)
        log_time_taken(transform_start, "Transforming S3 Data")

        # Step 5: Join S3 data with payment method
        join_start = time.time()
        joined_dyf = join_with_payment_method_id(s3_data_dyf, payment_method_dyf)
        log_time_taken(join_start, "Joining S3 Data with Payment Method Data")

        # Step 6: Write data to Redshift
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
