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
        
        column_mappings = [(name, name.lower()) for name in input_data.toDF().columns]

        # Rename columns using `rename_field`
        for old_name, new_name in column_mappings:
            input_data = input_data.rename_field(old_name, new_name)
        
        return input_data

    except Exception as e:
        # Log the error and re-raise the exception
        logger.error(f"Failed to load data from S3: {e}")
        logger.error(traceback.format_exc())
        raise e

# -----------------------------------
# Function to transform a specific column in a DynamicFrame
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
# Function to transform customer columns
# -----------------------------------

from datetime import datetime
from awsglue.dynamicframe import DynamicFrame

def transform_customer_columns_dyf(dyf):
    """
    Transform customer data by renaming and applying column-level transformations.

    Args:
        dyf (DynamicFrame): The input DynamicFrame containing raw customer data.

    Returns:
        DynamicFrame: A DynamicFrame with transformed customer data.

    Raises:
        RuntimeError: If an error occurs during the transformation process.
    """
    try:
        logger.info("Starting customer data transformation...")

        # Log schema and count of input DynamicFrame
        logger.info(f"Schema of input DynamicFrame: {[field.name for field in dyf.schema()]}")
        input_count = dyf.count()
        logger.info(f"Record count in input DynamicFrame: {input_count}")

        # Ensure the required columns exist
        required_fields = ["customerid", "gender", "seniorcitizen", "partner", "dependents", "tenure", "age", "location", "contract"]
        for field in required_fields:
            if field not in [f.name for f in dyf.schema()]:
                raise ValueError(f"Missing required field '{field}' in input DynamicFrame.")

        # Map transformations using the DynamicFrame API
        def transform_record(record):
            import re
            # Safely fetch and process 'customerID'
            raw_customer_id = record.get("customerid", "").strip()
            parts = raw_customer_id.split("-")
            customer_id = re.sub(r"[^a-zA-Z0-9]", "", parts[0]) if len(parts) > 0 else None
            customer_name = re.sub(r"[^a-zA-Z0-9 ]", "", parts[1]) if len(parts) > 1 else None

            # Apply transformations
            record["rawCustomerid"] = raw_customer_id
            record["customerid"] = customer_id
            record["customername"] = customer_name
            record["gender"] = record.get("gender", "").strip().lower()
            record["location"] = record.get("location", "").strip().lower()
            record["contract"] = record.get("contract", "").strip().lower()
            record["updated_on"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            return record

        # Apply mapping and select transformed fields
        transformed_dyf = dyf.map(transform_record).select_fields([
            "customerid", "customername", "gender", "seniorcitizen", 
            "partner", "dependents", "tenure", "age", "location", 
            "contract", "updated_on"
        ])

        # Log success message
        transformed_count = transformed_dyf.count()
        logger.info(f"Customer data transformation completed successfully. Total records in transformed DynamicFrame: {transformed_count}")
        
        # Log schemas for debugging
        logger.info(f"Customer DynamicFrame schema inside transform customer columns: {[field.name for field in transformed_dyf.schema()]}")
        return transformed_dyf

    except ValueError as ve:
        # Log and raise specific column validation errors
        logger.error(f"Validation error: {ve}")
        raise ve
    except Exception as e:
        # Log and re-raise unexpected errors
        logger.error(f"Error during customer data transformation: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to transform customer data: {e}")


    except ValueError as ve:
        # Log and raise specific column validation errors
        logger.error(f"Validation error: {ve}")
        raise ve
    except Exception as e:
        # Log and re-raise any unexpected errors
        logger.error(f"Error during customer data transformation: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to transform customer data: {e}")



# -----------------------------------
# Function to cast columns and fill default values
# -----------------------------------
def cast_and_fill_defaults_dyf(dyf):
    """
    Cast columns to appropriate types and fill null values with defaults based on Redshift table schema.

    Args:
        dyf (DynamicFrame): The input DynamicFrame containing customer data.

    Returns:
        DynamicFrame: A DynamicFrame with casted column types and default values for nulls.

    Raises:
        RuntimeError: If an error occurs during the casting or filling process.
    """
    try:
        logger.info("Starting to cast column types and fill default values...")

        # Preprocess data to handle type conversions
        def preprocess_record(record):
            # Safely convert columns to match expected types
            record["seniorcitizen"] = bool(int(record.get("seniorcitizen", "0")))  # Convert "0"/"1" to boolean
            record["partner"] = record.get("partner", "No").strip().lower() == "yes"  # Convert "Yes"/"No" to boolean
            record["dependents"] = record.get("dependents", "No").strip().lower() == "yes"  # Convert "Yes"/"No" to boolean
            record["tenure"] = int(record.get("tenure", "0")) if record.get("tenure", "").isdigit() else 0
            record["age"] = int(record.get("age", "0")) if record.get("age", "").isdigit() else 0
            return record

        # Apply preprocessing
        preprocessed_dyf = dyf.map(preprocess_record)

        # Log schema after preprocessing
        logger.info("Displaying the schema after preprocessing:")
        logger.info(f"Schema: {[field.name + ': ' + field.dataType.typeName() for field in preprocessed_dyf.schema()]}")
        logger.info("Displaying the top 5 rows after preprocessing:")
        preprocessed_dyf.toDF().show(5, truncate=False)

        # Apply mapping to match Redshift schema
        casted_dyf = preprocessed_dyf.apply_mapping([
            ("customerid", "string", "customerid", "string"),
            ("customername", "string", "customername", "string"),
            ("gender", "string", "gender", "string"),
            ("seniorcitizen", "boolean", "seniorcitizen", "boolean"),
            ("partner", "boolean", "partner", "boolean"),
            ("dependents", "boolean", "dependents", "boolean"),
            ("tenure", "int", "tenure", "int"),
            ("age", "int", "age", "int"),
            ("location", "string", "location", "string"),
            ("contract", "string", "contract", "string"),
            ("updated_on", "string", "updated_on", "timestamp")
        ])

        # Log schema after casting
        logger.info("Displaying the schema after casting:")
        logger.info(f"Schema: {[field.name + ': ' + field.dataType.typeName() for field in casted_dyf.schema()]}")
        logger.info("Displaying the top 5 rows after casting:")
        casted_dyf.toDF().show(5, truncate=False)

        # Fill null values with defaults
        def fill_defaults(record):
            record["seniorcitizen"] = record.get("seniorcitizen", False)
            record["partner"] = record.get("partner", False)
            record["dependents"] = record.get("dependents", False)
            record["tenure"] = record.get("tenure", 0)
            record["age"] = record.get("age", 0)
            return record

        transformed_dyf = casted_dyf.map(fill_defaults)

        # Log success message and schema for debugging
        transformed_count = transformed_dyf.count()
        logger.info("Casting and default value filling completed successfully.")
        logger.info(f"Total records in transformed DynamicFrame: {transformed_count}")
        logger.info(f"Customer DynamicFrame schema after casting and default value filling: {[field.name for field in transformed_dyf.schema()]}")

        return transformed_dyf

    except Exception as e:
        # Log and raise any errors during the process
        logger.error(f"Error during casting and filling defaults: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to cast and fill default values: {e}")


def filter_new_records_dyf(transformed_dyf, existing_dyf, column_name):
    try:
        logger.info(f"Separating records based on column '{column_name}'...")

        # Drop unnecessary fields
        transformed_dyf = transformed_dyf.drop_fields(["contractid"])

        # Log record counts and schemas
        logger.info(f"Record count in transformed DynamicFrame: {transformed_dyf.count()}")
        logger.info(f"Schema of transformed DynamicFrame: {transformed_dyf.schema()}")

        # Validate column existence
        if column_name not in [field.name for field in transformed_dyf.schema()]:
            raise ValueError(f"Column '{column_name}' not found in transformed DynamicFrame.")

        if existing_dyf.count() == 0:
            logger.warning("Existing DynamicFrame is empty. All records will be treated as new.")
            empty_dyf = DynamicFrame.fromDF(
                SPARK.createDataFrame([], transformed_dyf.schema()),
                GLUE_CONTEXT,
                "empty_dyf"
            )
            return transformed_dyf, empty_dyf

        # Join operation
        joined_dyf = transformed_dyf.join(
            keys1=[column_name],
            keys2=[column_name],
            frame2=existing_dyf,
            transformation_ctx="joined_dyf"
        )

        # Filter new records
        new_records_dyf = joined_dyf.filter(
            lambda rec: rec.get(f"{column_name}_right") is None, transformation_ctx="new_records_dyf"
        )

        # Filter matched records
        matched_records_dyf = joined_dyf.filter(
            lambda rec: rec.get(f"{column_name}_right") is not None, transformation_ctx="matched_records_dyf"
        )

        # Update matched records
        updated_matched_dyf = matched_records_dyf.map(
            lambda rec: {key.replace("_left", ""): rec[key] for key in rec if key.endswith("_left")},
            transformation_ctx="updated_dyf"
        )

        # Clean up new records
        new_records_cleaned_dyf = new_records_dyf.map(
            lambda rec: {key.replace("_left", ""): rec[key] for key in rec if key.endswith("_left")},
            transformation_ctx="cleaned_dyf"
        )

        logger.info(f"New records count: {new_records_cleaned_dyf.count()}")
        logger.info(f"Matched and updated records count: {updated_matched_dyf.count()}")

        return new_records_cleaned_dyf, updated_matched_dyf

    except ValueError as ve:
        logger.error(f"Column validation error: {ve}")
        raise ve
    except Exception as e:
        logger.error(f"Error processing records based on column '{column_name}': {e}")
        logger.error(traceback.format_exc())
        raise

        raise RuntimeError(f"Error processing records: {e}")

# -----------------------------------
# Function to load dimension data from Redshift
# -----------------------------------

def load_dimension_data_dyf(table_name, key_column, value_column):
    """
    Load dimension data (e.g., location or contract) from Redshift using DynamicFrame.

    Args:
        table_name (str): The name of the Redshift table to load data from.
        key_column (str): The primary key column in the table.
        value_column (str): The value column to be processed.

    Returns:
        DynamicFrame: A DynamicFrame containing the key and value columns with processed values.

    Raises:
        RuntimeError: If an error occurs while loading data from Redshift.
    """
    try:
        logger.info(f"Loading dimension data from Redshift table '{table_name}'...")

        # Load data from Redshift as a DynamicFrame
        dimension_data_dyf = GLUE_CONTEXT.create_dynamic_frame.from_options(
            connection_type="redshift",
            connection_options={
                "url": REDSHIFT_CONNECTION_OPTIONS["url"],
                "user": REDSHIFT_CONNECTION_OPTIONS["user"],
                "password": REDSHIFT_CONNECTION_OPTIONS["password"],
                "dbtable": table_name,
                "redshiftTmpDir": REDSHIFT_TMP_DIR,  # Ensure this points to a valid S3 path
            },
            transformation_ctx="load_dimension_data"
        )

        # Log the schema of the loaded DynamicFrame
        logger.info(f"Loaded Dimension DynamicFrame schema: {[field.name for field in dimension_data_dyf.schema()]}")

        # Ensure key_column and value_column exist in the loaded DynamicFrame
        loaded_columns = [field.name for field in dimension_data_dyf.schema()]
        if key_column not in loaded_columns or value_column not in loaded_columns:
            raise ValueError(f"Required columns '{key_column}' or '{value_column}' not found in the dimension table schema. "
                             f"Available columns: {loaded_columns}")

        # Select the required key and value columns
        selected_dyf = dimension_data_dyf.select_fields([key_column, value_column])

        # Apply lowercase and trimming transformations using map
        def transform_record(record):
            record[value_column] = record[value_column].strip().lower() if record[value_column] else None
            return record

        transformed_dyf = selected_dyf.map(f=transform_record)

        # Log success message with record count
        record_count = transformed_dyf.count()
        logger.info(f"Successfully loaded and transformed {record_count} records from '{table_name}'.")

        # Log schema for debugging
        logger.info(f"Transformed Dimension DynamicFrame schema: {[field.name for field in transformed_dyf.schema()]}")

        return transformed_dyf

    except ValueError as ve:
        # Log specific validation errors
        logger.error(f"Validation error: {ve}")
        raise ve
    except Exception as e:
        # Log and raise any errors during the process
        logger.error(f"Error loading dimension data from Redshift table '{table_name}': {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load dimension data from '{table_name}': {e}")




# -----------------------------------
# Function to join customer data with a dimension
# -----------------------------------
def join_with_dimension_dyf(customer_dyf, dimension_dyf, customer_column, dimension_column, foreign_key_column):
    """
    Join customer data with a dimension and add a foreign key column using DynamicFrame.

    Args:
        customer_dyf (DynamicFrame): The DynamicFrame containing customer data.
        dimension_dyf (DynamicFrame): The dimension DynamicFrame to join with.
        customer_column (str): The column in the customer DynamicFrame to match.
        dimension_column (str): The column in the dimension DynamicFrame to match.
        foreign_key_column (str): The foreign key column to be added to the result.

    Returns:
        DynamicFrame: A DynamicFrame resulting from the join, with only the foreign key column retained.

    Raises:
        RuntimeError: If an error occurs during the join process.
    """
    try:
        logger.info(f"Joining customer data with dimension on '{customer_column}' and '{dimension_column}'...")

        # Validate and clean join keys
        customer_dyf = customer_dyf.filter(lambda record: record[customer_column] is not None) \
                                   .map(lambda record: {
                                       **record,
                                       customer_column: record[customer_column].strip().lower() if record[customer_column] else None
                                   })

        dimension_dyf = dimension_dyf.filter(lambda record: record[dimension_column] is not None) \
                                     .map(lambda record: {
                                         **record,
                                         dimension_column: record[dimension_column].strip().lower() if record[dimension_column] else None
                                     })

        # Perform the join operation
        joined_dyf = customer_dyf.join(
            paths1=[customer_column],
            paths2=[dimension_column],
            frame2=dimension_dyf,
            transformation_ctx="join_customer_with_dimension"
        )

        # Rename the foreign key column if necessary
        if f"{foreign_key_column}_right" in [field.name for field in joined_dyf.schema()]:
            joined_dyf = joined_dyf.rename_field(f"{foreign_key_column}_right", foreign_key_column)

        # Retain only necessary columns
        result_dyf = joined_dyf.drop_fields([customer_column, dimension_column])

        # Log distinct `locationId` values
        logger.info("Fetching distinct foreign key values (locationId) from the joined DynamicFrame...")
        distinct_foreign_keys = result_dyf.toDF().select(foreign_key_column).distinct().collect()
        logger.info(f"Distinct {foreign_key_column} values in the joined DynamicFrame: {[row[foreign_key_column] for row in distinct_foreign_keys]}")

        # Log record count and schema after join
        result_count = result_dyf.count()
        logger.info(f"Join operation completed successfully. Record count in result DynamicFrame: {result_count}")
        logger.info(f"Result DynamicFrame schema: {[field.name for field in result_dyf.schema()]}")

        # Show the top 5 records in the result DynamicFrame after the join
        logger.info("Displaying the top 5 records in the Result DynamicFrame after the join:")
        result_dyf.toDF().show(5, truncate=False)

        return result_dyf

    except Exception as e:
        logger.error(f"Error joining customer data with dimension on columns '{customer_column}' and '{dimension_column}': {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to join customer data with dimension: {e}")




# -----------------------------------
# Function to log and handle missing foreign keys in the data
# -----------------------------------
def log_missing_foreign_keys_dyf(dyf, key_columns):
    """
    Log and handle missing foreign keys in the data using DynamicFrame.

    Args:
        dyf (DynamicFrame): The input DynamicFrame to check for missing foreign keys.
        key_columns (list): A list of foreign key column names to validate.

    Raises:
        RuntimeError: If an error occurs during the validation process.
    """
    try:
        logger.info(f"Checking for missing foreign keys in columns: {', '.join(key_columns)}...")
        input_count = dyf.count()
        logger.info(f"Record count in input DynamicFrame: {input_count}")

        # Filter rows with null values in any of the specified foreign key columns
        def filter_missing_keys(record):
            return any(record.get(key) is None for key in key_columns)

        missing_keys_dyf = dyf.filter(filter_missing_keys)

        # Count and log missing foreign keys
        missing_count = missing_keys_dyf.count()
        if missing_count > 0:
            logger.warning(f"Missing foreign keys detected for columns {', '.join(key_columns)}. Total missing: {missing_count}")
            logger.info("Displaying rows with missing foreign keys:")
            # Convert to DataFrame for better visualization in logs (optional)
            missing_keys_dyf.toDF().show(truncate=False)
        else:
            logger.info(f"No missing foreign keys found in columns: {', '.join(key_columns)}.")

    except Exception as e:
        # Log and raise errors during the validation process
        logger.error(f"Error checking for missing foreign keys in columns: {', '.join(key_columns)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to validate foreign keys: {e}")




# -----------------------------------
# Function to process and transform customer details
# -----------------------------------
def process_customer_details_dyf(dyf):
    """
    Process and transform customer details, linking with location and contract details.

    Args:
        dyf (DynamicFrame): The input DynamicFrame containing customer details.

    Returns:
        DynamicFrame: A Glue DynamicFrame containing the transformed customer data.

    Raises:
        RuntimeError: If any step in the transformation process fails.
    """
    try:
        logger.info("Starting to process customer details and transform data...")

        # Step 1: Transform customer columns
        logger.info("Transforming customer columns...")
        customer_dyf = transform_customer_columns_dyf(dyf)
        logger.info("Customer columns transformed successfully.")
        
        logger.info(f"Final DynamicFrame schema before step 2 (cast and fill): {[field.name for field in customer_dyf.schema()]}")
        
        logger.info("Displaying the top 5 rows of the transformed DynamicFrame:")
        customer_dyf.toDF().show(5, truncate=False)  # Display the top 5 rows without truncation
        

        # Step 2: Cast columns and fill defaults
        logger.info("Casting columns to appropriate types and filling default values...")
        customer_dyf = cast_and_fill_defaults_dyf(customer_dyf)
        logger.info("Casting and default filling completed successfully.")
        
        logger.info(f"Final DynamicFrame schema after casting and fill default: {[field.name for field in customer_dyf.schema()]}")
        
        logger.info("Displaying the top 5 rows of the transformed DynamicFrame:")
        customer_dyf.toDF().show(5, truncate=False)  # Display the top 5 rows without truncation

        # Step 3: Load dimension data
        logger.info("Loading dimension data for location and contract...")
        location_dyf = load_dimension_data_dyf(
            table_name="public.dim_location_details",
            key_column="locationid",
            value_column="locationname"
        )
        contract_dyf = load_dimension_data_dyf(
            table_name="public.dim_contract_details",
            key_column="contractid",
            value_column="contractname"
        )
        logger.info("Dimension data loaded successfully.")

        # Step 4: Join with dimensions
        logger.info("Joining customer data with location and contract dimensions...")
        customer_dyf = join_with_dimension_dyf(
            customer_dyf=customer_dyf,
            dimension_dyf=location_dyf,
            customer_column="location",
            dimension_column="locationname",
            foreign_key_column="locationid"
        )
        customer_dyf = join_with_dimension_dyf(
            customer_dyf=customer_dyf,
            dimension_dyf=contract_dyf,
            customer_column="contract",
            dimension_column="contractname",
            foreign_key_column="contractid"
        )
        logger.info("Join operations completed successfully.")
        
        logger.info(f"Final DynamicFrame schema after joining dimension tables: {[field.name for field in customer_dyf.schema()]}")
        
        logger.info("Displaying the top 5 rows of the transformed DynamicFrame:")
        customer_dyf.toDF().show(5, truncate=False)  # Display the top 5 rows without truncation

        # Step 5: Log missing foreign keys
        logger.info("Checking for missing foreign keys in locationId and contractId...")
        log_missing_foreign_keys_dyf(customer_dyf, ["locationid", "contractid"])
        logger.info("Missing foreign key validation completed.")

        # Step 6: Return the transformed DynamicFrame
        logger.info(f"Customer data transformation completed. Total records: {customer_dyf.count()}")
        
        logger.info(f"Final DynamicFrame schema afteer checking Foreign keys: {[field.name for field in customer_dyf.schema()]}")
        
        logger.info("Displaying the top 5 rows of the transformed DynamicFrame:")
        customer_dyf.toDF().show(5, truncate=False)  # Display the top 5 rows without truncation
        return customer_dyf

    except Exception as e:
        # Log and raise errors during the transformation process
        logger.error(f"Error processing customer details: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to process customer details: {e}")

# -----------------------------------
# Function to write customer contract details to Redshift with append only
# -----------------------------------
def write_customer_contract_dyf(customer_dynamic_frame):
    try:
        customer_contract_data_dyf = customer_dynamic_frame.select_fields(["customerid", "contractid"])
        if customer_contract_data_dyf.count() > 0:
            logger.info("Inserting new customer records into Redshift...")
            insert_new_records_to_redshift_dyf(
                new_records_dyf=customer_contract_data_dyf,
                table_name="public.customer_contract_details",
                transformation_ctx="Redshift_New_Customers_Contract_Insert"
            )
            logger.info("Customer Contract details successfully written to Redshift.")
        else:
            logger.info("No new customer records to insert into Redshift.")

    except Exception as e:
        # Log and raise any errors during the process
        logger.error(f"Error writing customer details to Redshift: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to write customer details to Redshift: {e}")
    

# -----------------------------------
# Function to write customer details to Redshift with incremental load
# -----------------------------------
def write_customer_details_dyf(customer_dynamic_frame):
    """
    Write customer details into Redshift with incremental load using DynamicFrame.

    Args:
        customer_dynamic_frame (DynamicFrame): The transformed customer data as a Glue DynamicFrame.

    Raises:
        RuntimeError: If any step in the incremental load process fails.
    """
    try:
        logger.info("Starting to write customer contract details to Redshift table....")
        write_customer_contract_dyf(customer_dynamic_frame)

        # Step 2: Load existing customer records from Redshift
        logger.info("Loading existing customer records from Redshift...")
        existing_customers_dyf = load_existing_data_from_redshift_dyf(
            table_name="public.customer_details",
            column_name="customerid"
        )
        logger.info(f"Successfully loaded existing customer records. Total records: {existing_customers_dyf.count()}")

        # Step 2: Filter out existing records
        logger.info("Filtering out existing customer records...")
        new_records_dyf, update_records_dyf = filter_new_records_dyf(
            transformed_dyf=customer_dynamic_frame,
            existing_dyf=existing_customers_dyf,
            column_name="customerid"
        )
        logger.info(f"Filtered new customer records. Total new records: {new_customers_dyf.count()}")

        # Step 3: Insert new records into Redshift
        if new_records_dyf.count() > 0:
            logger.info("Inserting new customer records into Redshift...")
            insert_new_records_to_redshift_dyf(
                new_records_dyf=new_records_dyf,
                table_name="public.customer_details",
                transformation_ctx="Redshift_New_Customers_Insert"
            )
            logger.info("Customer details successfully written to Redshift.")
        else:
            logger.info("No new customer records to insert into Redshift.")
            
        #step 4: Update the exsiting records in Redshift Customer_Details table
        
        # Generate the VALUES clause for the update query

        
        if updated_records_dyf.count() > 0:
            # Create the SQL query
            preactions_sql = f"""
            BEGIN;
            
            -- Update existing rows in customer_details
            UPDATE public.customer_details
            SET customername = new_data.customername,
                gender = new_data.gender,
                seniorcitizen = new_data.seniorcitizen,
                partner = new_data.partner,
                dependents = new_data.dependents,
                tenure = new_data.tenure,
                age = new_data.age,
                locationid = new_data.locationid,
                updated_on = new_data.updated_on
            FROM (
                SELECT * FROM (VALUES
                    {values_clause}
                ) AS temp_table(customerid, customername, gender, seniorcitizen, partner, dependents, tenure, age, locationid, updated_on)
            ) AS new_data
            WHERE public.customer_details.customerid = new_data.customerid;
            
            COMMIT;
            """
            
            
            # Create an empty DynamicFrame
            empty_dyf = DynamicFrame.fromDF(
                SparkSession.builder.getOrCreate().createDataFrame([], update_records_dyf.schema()), 
                glueContext
            )
            
            # Execute the update query using preactions, but don't append data
            GLUE_CONTEXT.write_dynamic_frame.from_options(
                frame=empty_dyf,  # Pass an empty DynamicFrame to avoid appending
                connection_type="redshift",
                connection_options={
                    **REDSHIFT_CONNECTION_OPTIONS,
                    "dbtable": "public.customer_details",  # Target table
                    "preactions": preactions_sql          # SQL for updating records
                },
                transformation_ctx="update_records"
            )
        
        


    except Exception as e:
        # Log and raise any errors during the process
        logger.error(f"Error writing customer details to Redshift: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to write customer details to Redshift: {e}")



try:
    logger.info("Starting ETL job...")

    # Step 1: Load raw data from S3
    logger.info("Loading raw data from S3...")
    s3_dynamic_frame = load_data_from_s3()
    logger.info(f"Total records loaded from S3: {s3_dynamic_frame.count()}")


    # Step 2: Process customer details
    logger.info("Processing customer details...")
    customer_dynamic_frame = process_customer_details_dyf(s3_dynamic_frame)
    logger.info("Customer details processed successfully.")

    # Step 3: Write customer details to Redshift
    logger.info("Writing customer details to Redshift...")
    write_customer_details_dyf(customer_dynamic_frame)
    logger.info("Customer details written to Redshift successfully.")

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

