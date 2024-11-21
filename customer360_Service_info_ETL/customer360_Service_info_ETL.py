import sys
import boto3
from botocore.exceptions import ClientError
import json
import logging
import time
from awsglue.transforms import *
from awsgluedq.transforms import EvaluateDataQuality
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from datetime import datetime
from awsglue.dynamicframe import DynamicFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from awsglue.transforms import ApplyMapping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Glue job context
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'TempDir'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
logger.info("ETL job initialized successfully.")

def log_time(func):
    """Decorator to measure and log the execution time of functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} completed in {elapsed_time:.2f} seconds.")
        return result
    return wrapper

@log_time
def load_raw_data():
    """
    Loads raw data from an S3 path into a DynamicFrame without applying mappings.
    Uses AWS Glue Job Bookmarks for incremental data processing.

    Args:
        None

    Returns:
        DynamicFrame: A DynamicFrame containing the raw customer data from S3.

    Raises:
        Exception: If an error occurs during data loading.
    """
    try:
        logger.info("Loading data from S3...")
 
        # S3 path to the data folder
        s3_path = "s3://cutomer360-s3-sample/raw/" 
    
        # Load the data from S3 using the create_dynamic_frame.from_options method
        raw_data = glueContext.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={
                "paths": [s3_path],
                "recurse": True,
                "groupFiles": "inPartition",
                "jobBookmarkKeys": ["filename"],  # Use filename as the bookmark key
                "jobBookmarkKeysSortOrder": "asc"
            },
            transformation_ctx="Redshift_job_bookmark_service_details",
            format="csv",  # Use "csv" format
            format_options={
                "withHeader": True,
                "separator": ",",  
                "quoteChar": '"'  
            }
        )
    
        # Check if any records were loaded
        record_count = raw_data.count()
        logger.info("Data loading from s3 is successful.")
        logger.info(f"Record count: {record_count}")

        if record_count == 0:
            logger.warning("No data found in the specified S3 path. Stopping ETL process.")
            return None  # Return None to indicate no data was processed
        return raw_data
    
        
    except Exception as e:
        logger.error(f"Error loading data from S3: {e}")
        raise
    
@log_time
def apply_mappings(raw_data):
    """
    Applies column mappings to the raw data to standardize column names and types.

    Args:
        raw_data (DynamicFrame): The input raw data.

    Returns:
        DynamicFrame: A DynamicFrame with standardized column names and types.

    Raises:
        Exception: If an error occurs during mapping.
    """
    try:
        logger.info("Applying column mappings...")
        mapped_data = ApplyMapping.apply(
            frame=raw_data,
            mappings=[
                ("customerID", "string", "customer_id", "string"),
                ("gender", "string", "gender", "string"),
                ("SeniorCitizen", "string", "senior_citizen", "int"),
                ("Partner", "string", "partner", "string"),
                ("Dependents", "string", "dependents", "string"),
                ("tenure", "string", "tenure", "int"),
                ("PhoneService", "string", "PhoneService", "string"),
                ("MultipleLines", "string", "MultipleLines", "string"),
                ("InternetService", "string", "InternetService", "string"),
                ("OnlineSecurity", "string", "OnlineSecurity", "string"),
                ("OnlineBackup", "string", "OnlineBackup", "string"),
                ("DeviceProtection", "string", "DeviceProtection", "string"),
                ("TechSupport", "string", "TechSupport", "string"),
                ("StreamingTV", "string", "StreamingTV", "string"),
                ("StreamingMovies", "string", "StreamingMovies", "string"),
                ("Contract", "string", "contract", "string"),
                ("PaperlessBilling", "string", "paperless_billing", "string"),
                ("PaymentMethod", "string", "payment_method", "string"),
                ("MonthlyCharges", "string", "monthly_charges", "decimal(10,2)"),
                ("TotalCharges", "string", "total_charges", "decimal(10,2)"),
                ("Churn", "string", "churn", "string"),
                ("Location", "string", "location", "string"),
                ("DataUsage_MB", "string", "data_usage", "decimal(10,2)"),
                ("CallMinutes", "string", "call_minutes", "decimal(10,2)"),
                ("Age", "string", "age", "int")
            ]
        )
        logger.info("Column mappings applied successfully.")
        return mapped_data

    except Exception as e:
        logger.error(f"Error applying column mappings: {e}")
        raise

    
def get_secret():
    """
    Connects to AWS Secrets Manager, retrieves the secret string containing 
    Redshift connection credentials, and returns the credentials as a JSON object.
    
    Args:
        None

    Returns:
        dict: A dictionary containing Redshift connection details 
              such as username, password, host, and database name.
    Raises:
        ClientError: If there is an error accessing the secret, such as missing permissions
                     or if the secret is not found.
        Exception: For other exceptions during secret retrieval.
    """
    secret_name = "customer360_redshift_secrets"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # Handle specific exceptions
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            print("Secrets Manager can't decrypt the protected secret text using the provided KMS key.")
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"The requested secret {secret_name} was not found.")
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            print(f"The request was invalid due to: {e.response['Error']['Message']}")
        else:
            print(f"An error occurred: {e}")
        raise e
    else:
        # Parse the secret
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            secret = base64.b64decode(get_secret_value_response['SecretBinary']).decode('utf-8')

        return json.loads(secret)
        
# Fetching the secret_data and save it to the redshift_options
secret_data = get_secret()
redshift_options = secret_data


def evaluate_data_quality(dynamic_frame, ruleset, context_name):
    """
    Evaluates data quality using the EvaluateDataQuality transform.

    Args:
        dynamic_frame (DynamicFrame): The input DynamicFrame to evaluate.
        ruleset (str): Data Quality ruleset in DQDL (Data Quality DSL) format.
        context_name (str): Name for the data quality evaluation context (used in logs and metrics).

    Returns:
        dict: Results of the data quality evaluation, including overall status.

    Raises:
        Exception: If an error occurs during data quality evaluation.
    """
    try:
        logger.info(f"Evaluating data quality for context: {context_name}...")

        # Apply data quality rules
        dq_results = EvaluateDataQuality().process_rows(
            frame=dynamic_frame,
            ruleset=ruleset,
            publishing_options={
                "dataQualityEvaluationContext": context_name,
                "enableDataQualityCloudWatchMetrics": True,
                "enableDataQualityResultsPublishing": True
            },
            additional_options={
                "observations.scope": "ALL",
                "performanceTuning.caching": "CACHE_NOTHING"
            }
        )

        logger.info(f"Data Quality Results for {context_name} has completed")
        return dq_results

    except Exception as e:
        logger.error(f"Error during data quality evaluation for {context_name}: {e}")
        raise
@log_time
def load_existing_data_from_redshift(table_name):
    """
    Load data from a specified Redshift table and return appropriate output based on the table.

    Args:
        table_name (str): The name of the Redshift table to load. 
                          Accepts "service_usage" or "dim_services".

    Returns:
        tuple: For "service_usage" table, returns a tuple of:
                - existing_ids (DynamicFrame): DynamicFrame of distinct `customer_id`s.
                - max_id (int): Maximum `service_usage_id`.
                - has_dim_service_data (bool): Indicates if there is data in the `service_usage` table.
        DynamicFrame: For "dim_services" table, returns the complete `dim_services` DynamicFrame.

    Raises:
        ValueError: If an unsupported table name is provided.
        Exception: For any errors during data loading from Redshift.
    """
    try:
        logger.info(f"Loading data from Redshift table: {table_name}...")

        # Load the specified Redshift table as a DynamicFrame
        redshift_dyf = glueContext.create_dynamic_frame.from_options(
            connection_type="redshift",
            connection_options={
                **redshift_options,
                "dbtable": f"public.{table_name}"
            },
            transformation_ctx=f"load_{table_name}"
        )

        # Count records to check for data presence
        record_count = redshift_dyf.count()
        has_dim_service_data = record_count > 0
        logger.info(f"Loaded {record_count} records from {table_name}.")

        if table_name == "service_usage":
            # Select distinct customer IDs directly in DynamicFrame
            existing_ids = redshift_dyf.select_fields(["customer_id"])

            # Convert to DataFrame temporarily for the aggregation needed to find max_id
            max_id_df = redshift_dyf.select_fields(["service_usage_id"]).toDF()
            max_id_row = max_id_df.agg({"service_usage_id": "max"}).collect()[0]
            max_id = max_id_row["max(service_usage_id)"] if max_id_row["max(service_usage_id)"] else 0

            logger.info(f"Distinct customer_id count in {table_name}: {existing_ids.count()}")
            logger.info(f"Current maximum service_usage_id in {table_name}: {max_id}")
            
            # Return DynamicFrame of existing_ids, max_id, and data presence flag
            return existing_ids, max_id, has_dim_service_data

        elif table_name == "dim_services":
            # Return the complete `dim_services` DynamicFrame
            return redshift_dyf

        else:
            logger.error(f"Table {table_name} is not recognized for specific transformations.")
            raise ValueError(f"Unsupported table {table_name} for this function.")

    except Exception as e:
        logger.error(f"Error loading data from Redshift table {table_name}: {str(e)}")
        raise
    

@log_time
def transform_dim_services(service_columns):
    """
    This function takes a list of service columns and creates a standardized `dim_services`
    table that assigns each service a unique `service_id` and includes its service name.
    The result is used to map services consistently across the ETL pipeline.

    Args:
        service_columns (list of str): A list of column names representing service types 
                                       (e.g., 'PhoneService', 'MultipleLines').
    Returns:
        DynamicFrame: A DynamicFrame representing the `dim_services` table, containing 
                      columns `service_id` (int) and `service_name` (str).
    Raises:
        ValueError: If `service_columns` is empty or None.
        Exception: For general errors during transformation.

    """
    try:
        logger.info("Transforming data for dim_services table...")

        # Ensure service_columns is not empty or None
        if not service_columns:
            raise ValueError("Service columns are empty or None.")

        # Transform service columns into a list of tuples
        service_data = [(i + 1, service) for i, service in enumerate(service_columns)]

        # Create a DataFrame from the transformed service data
        service_df = spark.createDataFrame(service_data, ["service_id", "service_name"])

        # Apply explicit casting to align with the Redshift table schema
        service_df = service_df.withColumn("service_id", F.col("service_id").cast("int"))
        service_df = service_df.withColumn("service_name", F.col("service_name").cast("string"))

        # Convert the DataFrame to a DynamicFrame
        dim_services_dyf = DynamicFrame.fromDF(service_df, glueContext, "dim_services_dyf")

        logger.info("dim_services table transformation complete.")
        return dim_services_dyf

    except ValueError as ve:
        # Specific error for invalid service_columns
        logger.error(f"ValueError during dim_services transformation: {str(ve)}")
        raise  # Reraise the exception after logging it

    except Exception as e:
        # General error handling for other exceptions
        logger.error(f"Error during dim_services transformation: {str(e)}")
        raise  # Reraise the exception after logging it
    
@log_time
def transform_service_usage(input_data, max_id):
    """
    This function takes the input customer data, selects relevant columns, and assigns 
    a unique `service_usage_id` to each record, starting from the provided `max_id`.
    It also adds an `updated_on` timestamp.

    Args:
        input_data (DynamicFrame): The DynamicFrame containing customer data from S3.
        max_id (int): The maximum current ID from Redshift's `service_usage` table to start
                      generating unique IDs for new records.

    Returns:
        DynamicFrame: A DynamicFrame for the `service_usage` table with a unique `service_usage_id`
                      and a timestamp column `updated_on`.

    Raises:
        ValueError: If `input_data` is None or missing required columns.
        Exception: For any errors during transformation.
    """
    try:
        logger.info("Starting transformation for service_usage table...")
        logger.info(f"max_id {max_id}")

        # Step 1: Select only the required columns from the input data
        logger.info("Selecting relevant columns for transformation.")
        selected_data = input_data.select_fields([
            "customer_id", "data_usage", "call_minutes", 
            "PhoneService", "MultipleLines", "InternetService", 
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
            "TechSupport", "StreamingTV", "StreamingMovies"
        ])

        # Convert DynamicFrame to DataFrame for transformation
        selected_df = selected_data.toDF()
        window_spec = Window.orderBy(F.lit(1))  # Order all rows the same way to generate sequential IDs

        # Step 2: Add `service_usage_id` and `updated_on`, and cast columns
        logger.info("Transforming and casting columns for service_usage...")
        transformed_df = (selected_df
            .withColumn("service_usage_id", F.row_number().over(window_spec) + max_id)  # Generate unique IDs
            .withColumn("customer_id", F.col("customer_id").cast("string"))             # Cast customer_id to string
            .withColumn("data_usage", F.col("data_usage").cast("decimal(10, 2)"))      # Cast data_usage to decimal
            .withColumn("call_minutes", F.col("call_minutes").cast("decimal(10, 2)"))  # Cast call_minutes to decimal
            .withColumn("updated_on", F.current_timestamp())                           # Add current timestamp
        )

        # Log the range of `service_usage_id` for debugging
        min_id = transformed_df.agg(F.min("service_usage_id")).collect()[0][0]
        max_generated_id = transformed_df.agg(F.max("service_usage_id")).collect()[0][0]
        logger.info(f"Generated service_usage_id range: Min={min_id}, Max={max_generated_id}")

        # Convert back to DynamicFrame
        transformed_dyf = DynamicFrame.fromDF(transformed_df, glueContext, "transformed_service_usage_dyf")

        logger.info("service_usage table transformation complete.")
        return transformed_dyf

    except ValueError as ve:
        logger.error(f"ValueError during service_usage transformation: {str(ve)}")
        raise  # Reraise the exception after logging it

    except Exception as e:
        logger.error(f"Error during service_usage transformation: {str(e)}")
        raise  # Reraise the exception after logging it
@log_time
def transform_service_usage_services(service_usage_dyf, service_columns, dim_services_df):
    """
    Transform data for service_usage_services table.
    Maps service_usage_id to service_id based on dim_services.
    Includes error handling to manage issues such as missing columns or invalid data.
    """
    try:
        logger.info("Transforming data for service_usage_services table...")

        # Validate inputs
        if service_usage_dyf is None or service_columns is None or dim_services_df is None:
            raise ValueError("One or more input parameters are None: service_usage_dyf, service_columns, or dim_services_df")

        # Convert DynamicFrame to DataFrame
        service_usage_df = service_usage_dyf.toDF()

        # Check if the required columns exist in service_usage_df
        missing_columns = [col for col in service_columns if col not in service_usage_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in service_usage_df: {', '.join(missing_columns)}")

        # Explode service columns into rows for mapping
        exploded_df = (
            service_usage_df.select("service_usage_id", *service_columns)
            .withColumn("exploded_service", F.explode(F.array([
                F.struct(F.lit(service_name).alias("service_name"), F.col(service_name).alias("value"))
                for service_name in service_columns
            ])))
            .select("service_usage_id", "exploded_service.service_name", "exploded_service.value")
            .filter(~F.col("value").rlike("(?i)^No"))  # Exclude " No" values
        )

        # Ensure there is at least one valid row after filtering
        if exploded_df.count() == 0:
            raise ValueError("No valid data left after filtering 'No' values.")

        # Join with dim_services to map service_name to service_id
        service_usage_services_df = exploded_df.join(
            dim_services_df,
            exploded_df["service_name"] == dim_services_df["service_name"],
            how="inner"
        ).select(
            "service_usage_id", dim_services_df["service_id"]
        )

        # Ensure the join resulted in some data
        if service_usage_services_df.count() == 0:
            raise ValueError("The join between exploded data and dim_services returned no matching records.")

        # Convert back to DynamicFrame
        service_usage_services_dyf = DynamicFrame.fromDF(service_usage_services_df, glueContext, "service_usage_services_dyf")
        logger.info("service_usage_services table transformation complete.")
        return service_usage_services_dyf

    except ValueError as ve:
        # Handle specific value errors (e.g., missing columns, empty data after filtering, no join results)
        logger.error(f"ValueError during service_usage_services transformation: {str(ve)}")
        raise  # Reraise the exception after logging it

    except Exception as e:
        # General error handling for other unexpected errors
        logger.error(f"Error during service_usage_services transformation: {str(e)}")
        raise  # Reraise the exception after logging it
    
@log_time
def write_to_redshift(dyf, table_name):
    """
    This function takes a transformed DynamicFrame and writes it to the specified
    table in Redshift, using the Redshift connection options. It is designed to be
    reusable for different tables within the ETL pipeline.

    Args:
        dyf (DynamicFrame): The DynamicFrame to be written to Redshift.
        table_name (str): The name of the Redshift table to write the data into.
    Returns:
        None
    Raises:
        ValueError: If `dyf` is None or if `table_name` is not specified.
        Exception: For any errors during the write operation to Redshift.
    """
    try:
        logger.info(f"Writing {table_name} to Redshift...")
    
        # Pass redshift options dynamically and specify the dbtable
        glueContext.write_dynamic_frame.from_options(
            frame=dyf,
            connection_type="redshift",
            connection_options={
                **redshift_options,  # Unpack redshift_options dictionary
                "dbtable": f"public.{table_name}"  # Specify the table name dynamically
            },
            transformation_ctx=f"Redshift_{table_name}"
        )
        logger.info(f"{table_name} written successfully.")

    except ValueError as ve:
        # Handle missing input values (e.g., None)
        logger.error(f"ValueError: {str(ve)}")
        raise  # Reraise the exception after logging it

    except Exception as e:
        # Handle general errors (e.g., connection issues, permission issues)
        logger.error(f"Error writing {table_name} to Redshift: {str(e)}")
        raise  # Reraise the exception after logging it
    
@log_time
def main():
    """
    Main ETL function

    This function executes the ETL steps to load, transform, and store customer data. It performs:
    Steps:
        1. Load raw customer data from S3.
        2. Load existing `service_usage` data and max `service_usage_id` from Redshift.
        3. Load or transform `dim_services` table based on existing Redshift data.
        4. Transform customer data for `service_usage` table.
        5. Transform and map active services for `service_usage_services` table.
        6. Write transformed data to Redshift.

    Args:
        None
    Returns:
        None
    Raises:
        Exception: For any errors during the ETL pipeline execution.
    """
    logger.info("Starting ETL process...")
    raw_data = load_raw_data()
    if raw_data is None:
        logger.info("No raw input data found. Committing job and exiting.")
        job.commit()
        return
    
    mapped_data = apply_mappings(raw_data)

    # Fetch existing data and max_id from Redshift
    existing_ids, max_id, has_dim_service_data = load_existing_data_from_redshift('service_usage')

    # Define service columns based on data
    service_columns = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # If dim_services exists, skip it; otherwise, write it
    if not has_dim_service_data:
        logger.info("dim_services table is empty. Writing dim_services data...")
        dim_services_dyf = transform_dim_services(service_columns)

        # Define data quality rules for dim_services
        ruleset = """
        Rules = [
            IsPrimaryKey "service_id"
        ]
        """
        dq_results = evaluate_data_quality(dim_services_dyf,ruleset,"dim_services")
        ruleOutcomes = SelectFromCollection.apply(
            dfc=dq_results,
            key="rowLevelOutcomes",
            transformation_ctx="rowLevelOutcomes",
        )

        ruleOutcomes.toDF().show(truncate=False)
        logger.info("Data Quality completed")
        if ruleOutcomes:
            job.commit()
            return
        
        
        
                
       


        # Step 4: Evaluate Overall Data Quality Outcome
        # if dq_results["overallResult"] == "PASSED":
        #     logger.info("Data Quality checks passed for dim_services table.")
        #     # Step 5: Write Data to Redshift
            
        # else:
        #     logger.error("Data Quality checks failed for dim_services table. Exiting job.")
        #     logger.error(f"Failed Data Quality Results: {dq_results}")
        #     job.commit()
        #     return
        write_to_redshift(dim_services_dyf, "dim_services")
        dim_services_df = dim_services_dyf.toDF()

        
    else:
        logger.info("dim_services has data. Skipping dim_services.")
        dim_services_dyf = load_existing_data_from_redshift('dim_services')
        dim_services_df = dim_services_dyf.toDF()

    # Transform data for each table
    service_usage_dyf = transform_service_usage(mapped_data, max_id)
    # Remove service columns before writing service_usage table to Redshift
    service_usage_df = service_usage_dyf.toDF().select(
        "service_usage_id", "customer_id", "data_usage", "call_minutes", "updated_on"
    )
        
    service_usage_dyf_cleaned = DynamicFrame.fromDF(service_usage_df, glueContext, "service_usage_dyf_cleaned")
    service_usage_services_dyf = transform_service_usage_services(service_usage_dyf, service_columns, dim_services_df)

    # Write each table to Redshift
    write_to_redshift(service_usage_dyf_cleaned, "service_usage") 
    write_to_redshift(service_usage_services_dyf, "service_usage_services")

    logger.info("ETL job completed successfully.")
    job.commit()
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"ETL job failed: {e}")
        raise