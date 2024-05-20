from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions import col, trim

DATA_FOLDER = "data"
OUTPUT_PATH = "data/transformed_data.csv"

def create_spark_session():
    return SparkSession.builder.appName("SparkSQL").getOrCreate()

def read_data(spark):
    # Define the schema for the dataset
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", FloatType(), True),
        StructField("education", StringType(), True),
        StructField("education_num", FloatType(), True),
        StructField("marital_status", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("relationship", StringType(), True),
        StructField("race", StringType(), True),
        StructField("sex", StringType(), True),
        StructField("capital_gain", FloatType(), True),
        StructField("capital_loss", FloatType(), True),
        StructField("hours_per_week", FloatType(), True),
        StructField("native_country", StringType(), True),
        StructField("income", StringType(), True)
    ])

    # Read the dataset
    data = spark.read \
        .schema(schema) \
        .option("header", "false") \
        .option("inferSchema", "false") \
        .csv(f"{DATA_FOLDER}/*.csv")

    # Convert float columns to integer
    float_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, FloatType)]
    for v in float_columns:
        data = data.withColumn(v, data[v].cast(IntegerType()))

    # Trim leading and trailing spaces in string columns
    string_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, StringType)]
    for column in string_columns:
        data = data.withColumn(column, trim(data[column]))

    return data

def clean_data(data):
    # Drop rows with any null values
    return data.dropna()

def transform_data(data):
    # Create columns consisting of all products of integer columns
    integer_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, IntegerType)]
    for i, col1 in enumerate(integer_columns):
        for col2 in integer_columns[i:]:
            product_col_name = f"{col1}_x_{col2}"
            data = data.withColumn(product_col_name, col(col1) * col(col2))
    return data

def save_data(df, output_path):
    df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

def main():
    spark = create_spark_session()
    data = read_data(spark)
    data_cleaned = clean_data(data)
    data_transformed = transform_data(data_cleaned)
    save_data(data_transformed, OUTPUT_PATH)
    spark.stop()

if __name__ == "__main__":
    main()
