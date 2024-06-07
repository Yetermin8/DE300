from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import boto3
import tomli
import os
from io import StringIO
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sqrt, lit, udf
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
import requests
from scrapy import Selector
import re

# Read the parameters from toml
CONFIG_BUCKET = "de300spring2024-yetayaltizale-airflow"
CONFIG_FILE_KEY = "config_yetayaltizale_hw4.toml"

TABLE_NAMES = {
    "original_data": "heart_disease",
    "clean_data_pandas": "heart_disease_clean_data_pandas",
    "clean_data_spark": "heart_disease_clean_data_spark",
    "train_data_pandas": "heart_disease_train_data_pandas",
    "test_data_pandas": "heart_disease_test_data_pandas",
    "train_data_spark": "heart_disease_train_data_spark",
    "test_data_spark": "heart_disease_test_data_spark",
    "max_fe": "heart_disease_max_fe_features",
    "product_fe": "heart_disease_product_fe_features",
    "eda_results": "heart_disease_eda_results",
    "model_results": "heart_disease_model_results"
}

# Define the default args dictionary for DAG
default_args = {
    'owner': 'yetayaltizale',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

def read_config_from_s3() -> dict:
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=CONFIG_BUCKET, Key=CONFIG_FILE_KEY)
    config_data = obj['Body'].read().decode('utf-8')
    params = tomli.loads(config_data)
    return params

# Usage
PARAMS = read_config_from_s3()

def create_db_connection():
    """
    Create a database connection to the PostgreSQL RDS instance using SQLAlchemy.
    """
    conn_uri = f"{PARAMS['db']['db_alchemy_driver']}://{PARAMS['db']['username']}:{PARAMS['db']['password']}@{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}"
    engine = create_engine(conn_uri)
    return engine

def add_data_to_table_func():
    """
    Insert data from a CSV file stored in S3 to a database table.
    """
    engine = create_db_connection()
    conn = engine.connect()

    s3_bucket = PARAMS['files']['s3_bucket']
    s3_key = PARAMS['files']['s3_file_key']
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)

    csv_content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content))

    df.to_sql(TABLE_NAMES['original_data'], con=conn, if_exists="replace", index=False)
    conn.close()
    return {'status': 1}

def eda_func():
    engine = create_db_connection()
    conn = engine.connect()
    try:
        result = conn.execute(f"SELECT * FROM {TABLE_NAMES['original_data']}")
        rows = result.fetchall()
        column_names = result.keys()
        df = pd.DataFrame(rows, columns=column_names)

        # List of features with most null values to drop
        features_with_most_null_values = ['restckm', 'pncaden', 'earlobe', 'exerckm', 'exeref', 'exerwm', 'restef', 'restwm', 'thalpul', 'dm', 'thalsev']
        df.drop(columns=features_with_most_null_values, inplace=True, errors='ignore')

        # Missing values analysis
        missing_percentage = df.isnull().sum() * 100 / len(df)
        print("Missing values percentage per column:\n", missing_percentage)

        # Imputation for numerical features
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)  # using median to impute

        # Identify non-categorical variables that are numeric
        non_categorical_variables = [var for var in df.columns if pd.api.types.is_numeric_dtype(df[var])]

        # Handling outliers
        for column in non_categorical_variables:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

        # Correlation Analysis
        corr_matrix = df.corr()
        corr_flat = corr_matrix.unstack().reset_index()
        corr_flat.columns = ['Feature1', 'Feature2', 'Correlation']
        corr_flat = corr_flat[corr_flat['Feature1'] != corr_flat['Feature2']]
        corr_flat['Abs_Correlation'] = corr_flat['Correlation'].abs()
        sorted_corr = corr_flat.sort_values(by='Abs_Correlation', ascending=False).head(10)
        
        sorted_corr.to_sql(TABLE_NAMES['eda_results'], con=conn, if_exists="replace", index=False)
    except Exception as e:
        print(f"Error in EDA function: {e}")
        raise
    finally:
        conn.close()


def fe_1_func():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['original_data']}", conn)
    
    # Example FE: Adding interaction term
    df['age_trestbps'] = df['age'] * df['trestbps']
    
    df.to_sql(TABLE_NAMES['max_fe'], con=conn, if_exists="replace", index=False)
    conn.close()

def fe_2_func():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['original_data']}", conn)
    
    # Example FE: Adding squared term
    df['trestbps_squared'] = df['trestbps'] ** 2
    
    df.to_sql(TABLE_NAMES['product_fe'], con=conn, if_exists="replace", index=False)
    conn.close()

def model_training_lr_func():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['max_fe']}", conn)
    
    # Example model training with Logistic Regression
    X = df.drop('target', axis=1)
    y = df['target']
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results = pd.DataFrame({'model': ['Logistic Regression'], 'accuracy': [accuracy]})
    results.to_sql(TABLE_NAMES['model_results'], con=conn, if_exists="replace", index=False)
    conn.close()

def model_training_rf_func():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['max_fe']}", conn)
    
    # Example model training with Random Forest
    X = df.drop('target', axis=1)
    y = df['target']
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results = pd.DataFrame({'model': ['Random Forest'], 'accuracy': [accuracy]})
    results.to_sql(TABLE_NAMES['model_results'], con=conn, if_exists="replace", index=False)
    conn.close()

def model_training_gbt_func():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['max_fe']}", conn)
    
    # Example model training with Gradient Boosting
    X = df.drop('target', axis=1)
    y = df['target']
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results = pd.DataFrame({'model': ['Gradient Boosting'], 'accuracy': [accuracy]})
    results.to_sql(TABLE_NAMES['model_results'], con=conn, if_exists="replace", index=False)
    conn.close()

def compare_models():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['model_results']}", conn)
    
    best_model = df.loc[df['accuracy'].idxmax()]
    print(f"The best model is {best_model['model']} with accuracy {best_model['accuracy']}")
    
    conn.close()
    
def create_spark_session():
    spark = SparkSession.builder \
        .appName("Heart Disease Analysis HW4") \
        .getOrCreate()
    return spark

def spark_clean_data_func():
    spark = SparkSession.builder \
        .appName("Heart Disease Analysis HW4") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .getOrCreate()

    # Set the AWS credentials explicitly if necessary
    hadoop_conf = spark._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.access.key", os.environ.get("ASIAYAAO5HRMMCFQWV74"))
    hadoop_conf.set("fs.s3a.secret.key", os.environ.get("XFqCtrcVnzZwA2w9XA2d0IRWmcloDl0RvxoXWKRU"))
    hadoop_conf.set("fs.s3a.session.token", os.environ.get("IQoJb3JpZ2luX2VjEG0aCXVzLWVhc3QtMiJIMEYCIQCoeJgOLkvK+LMH/W2rj8bOpvD0yGi1+4GsMu9AwJMCVwIhAN1g6e8cWSvLHKhwbqHNZeJwUGq0kVqpZw7jwdjp5GEAKvQCCPb//////////wEQABoMNTQ5Nzg3MDkwMDA4IgzB8rT6fWDLOWdpvFMqyAJ2AqB+5ibBUwhINaYApZ6KczRc6GXQzh8Oo4ykSaQW1bQOxm6lM5ayfrs01Y71OKBfpzINeOMvSrBRo17rcCy9GDEyJavcS+ICSk+DUt+P1Ma5y3AmEKe4tQjB7IgIr0roPML6fu+3Gjjto7Uwh5UrIyolxFyjkEqm6Wdr6M1DyIqJpXCaz3/jdClOHlZPFuFA431uPaxk8ovaBgoX2vbtEso2OpsB8wJBCSOGnN++JZ0bHjevEt5m7/uh4b6lBzEE7ZpEkAZ8T6MxnPKbu+Z5K7Ll8KF5ywf8cenq8HJAPy1t5jg8m5alN+6gOBmK0QfYSd86I/+Xge0gxMcAkgQbJlEXHg33j5teErN+HhtavWJGMREUsRMOAPzZpISw7xm4a+0kYiH7jdYRT1ZwnUvyG1XKAhKVJoaM0IZV/uKYUuDfk7DM9XV4MOrzjbMGOqYBzX3aO69qISbDFQ0onZssyKMughv4KDDGGkKB0JPwehHfeWXHe7uS5kuTgnUBUrsd3e2+EFPy/o4JvsffWUEQcgw6bmqMwdP1pun0tlDM/fBT2O+5tHJ4tmgCX0HDfjDahpmuijpO7x9734Z6ENut22gxV2xcL4S41YKE9UJU+YTIEoCs3HT5eiY0EpNfUMwsCSMgBaRWlFCxHW7ylJy5RDEpGbO0uw=="))

    df = spark.read.csv("s3://de300spring2024-yetayaltizale-airflow/heart_disease.csv", header=True, inferSchema=True)

    # Cleaning steps
    df = df.limit(899)
    columns_to_keep = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df = df.select(*columns_to_keep)

    binary_cols = ['painloc', 'painexer', 'exang']
    for col_name in binary_cols:
        mode_value = df.groupby(col_name).count().orderBy('count', ascending=False).first()[0]
        df = df.fillna({col_name: mode_value})
    
    median_trestbps = df.approxQuantile('trestbps', [0.5], 0.01)[0]
    df = df.withColumn('trestbps', when(df['trestbps'] < 100, median_trestbps).otherwise(df['trestbps']))
    df = df.na.fill({'trestbps': median_trestbps})

    median_oldpeak = df.approxQuantile('oldpeak', [0.5], 0.01)[0]
    df = df.withColumn('oldpeak', when((df['oldpeak'] < 0) | (df['oldpeak'] > 4), median_oldpeak).otherwise(df['oldpeak']))
    df = df.na.fill({'oldpeak': median_oldpeak})

    for col_name in ['thaldur', 'thalach']:
        median_value = df.approxQuantile(col_name, [0.5], 0.01)[0]
        df = df.na.fill({col_name: median_value})

    binary_cols = ['fbs', 'prop', 'nitr', 'pro', 'diuretic']
    for col_name in binary_cols:
        df = df.withColumn(col_name, when(df[col_name] > 1, 1).otherwise(df[col_name]))
        df = df.na.fill({col_name: 0})

    mode_slope = df.groupby('slope').count().orderBy('count', ascending=False).first()[0]
    df = df.na.fill({'slope': mode_slope})
    df = df.withColumn('age', col('age').cast('float'))
    df = df.filter((df['age'] >= 18) & (df['age'] < 75))

    # Function to fetch ABS smoking rates
    def fetch_smoking_rates_abs():
        url = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release"
        response = requests.get(url)
        if response.status_code == 200:
            selector = Selector(text=response.text)
            rows = selector.xpath('//table[caption[contains(text(), "Proportion of people 15 years and over who were current daily smokers by age and sex, 2022")]]/tbody/tr')
            return_dict = {}
            for row in rows:
                age_group = row.xpath('.//th/text()').get(default="N/A").strip()
                males_percentage = row.xpath('.//td[1]/text()').get(default="N/A").strip()
                females_percentage = row.xpath('.//td[4]/text()').get(default="N/A").strip()
                return_dict[age_group] = {"Males": float(males_percentage)/100, 'Females': float(females_percentage)/100}
            return return_dict
        else:
            print("Failed to retrieve data. Status code:", response.status_code)
            return {}

    # Function to fetch CDC smoking rates
    def fetch_smoking_rates_cdc():
        url = "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm"
        response = requests.get(url)
        if response.status_code == 200:
            selector = Selector(text=response.text)
            male_rate_text = selector.xpath("//li[contains(text(), 'adult men')]/text()").get()
            female_rate_text = selector.xpath("//li[contains(text(), 'adult women')]/text()").get()
            male_rate = float(re.search(r"(\d+\.\d+)", male_rate_text).group(1))
            female_rate = float(re.search(r"(\d+\.\d+)", female_rate_text).group(1))
            correction_factor = male_rate / female_rate if female_rate != 0 else 1
            age_smoking_rates = {}
            age_groups = selector.xpath("//div[h4[contains(text(), 'By Age')]]/following-sibling::div//ul/li")
            for group in age_groups:
                details = group.xpath(".//text()").get()
                age_range_match = re.search(r"(\d+[\–\-]\d+ years|\d+ years and older)", details)
                if age_range_match:
                    age_range = age_range_match.group(1)
                    age_range_rate = float(re.search(r"(\d+\.\d+)", details).group(1))
                    adjusted_rate = age_range_rate * correction_factor
                    age_smoking_rates[age_range] = {'female_rate': age_range_rate/100, 'male_rate': adjusted_rate/100}
            return age_smoking_rates
        else:
            print("Failed to retrieve data. Status code:", response.status_code)
            return {}

    # Fetch rates
    abs_rates = fetch_smoking_rates_abs()
    cdc_rates = fetch_smoking_rates_cdc()

    # Define mappings from your DataFrame's age groups to the ABS and CDC age groups
    def map_age_to_abs_group(age):
        if age < 18:
            return None
        elif 18 <= age <= 24:
            return '18–24'
        elif 25 <= age <= 34:
            return '25–34'
        elif 35 <= age <= 44:
            return '35–44'
        elif 45 <= age <= 54:
            return '45–54'
        elif 55 <= age <= 64:
            return '55–64'
        elif 65 <= age <= 74:
            return '65–74'
        elif age >= 75:
            return '75 years and over'
        else:
            return None

    def map_age_to_cdc_group(age):
        if age < 18:
            return None
        elif 18 <= age <= 24:
            return '18–24 years'
        elif 25 <= age <= 44:
            return '25–44 years'
        elif 45 <= age <= 64:
            return '45–64 years'
        elif age >= 65:
            return '65 years and older'
        else:
            return None

    # Function to get the smoking rate based on age group, sex and dataset
    def get_smoking_rate(age, sex, dataset):
        if dataset == 'abs':
            age_group = map_age_to_abs_group(age)
            sex_key = 'Males' if sex == 1 else 'Females'
            return abs_rates.get(age_group, {}).get(sex_key, None)
        elif dataset == 'cdc':
            age_group = map_age_to_cdc_group(age)
            sex_key = 'male_rate' if sex == 1 else 'female_rate'
            return cdc_rates.get(age_group, {}).get(sex_key, None)
        return None

    # Apply the functions and create new columns for ABS and CDC rates
    get_smoking_rate_udf_abs = F.udf(lambda age, sex: get_smoking_rate(age, sex, 'abs'), FloatType())
    get_smoking_rate_udf_cdc = F.udf(lambda age, sex: get_smoking_rate(age, sex, 'cdc'), FloatType())

    df = df.withColumn("abs_smoke_rate", get_smoking_rate_udf_abs(col("age"), col("sex")))
    df = df.withColumn("cdc_smoke_rate", get_smoking_rate_udf_cdc(col("age"), col("sex")))

    # Function to get the average smoking rate based on ABS and CDC
    def get_average_smoking_rate(age, sex):
        abs_rate = get_smoking_rate(age, sex, 'abs')
        cdc_rate = get_smoking_rate(age, sex, 'cdc')
        if abs_rate is not None and cdc_rate is not None:
            return (abs_rate + cdc_rate) / 2
        elif abs_rate is not None:
            return abs_rate
        elif cdc_rate is not None:
            return cdc_rate
        return None

    # UDF to apply average smoking rate imputation
    get_average_smoking_rate_udf = F.udf(lambda age, sex: get_average_smoking_rate(age, sex), FloatType())

    # Impute missing smoke values using the average smoking rate
    df = df.withColumn('smoke', when(df['smoke'].isNull(), get_average_smoking_rate_udf(col('age'), col('sex'))).otherwise(df['smoke']))

    # Check for null values in each column
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    null_counts.show()

    # Save cleaned data to SQL
    engine = create_db_connection()
    pandas_df = df.toPandas()
    pandas_df.to_sql(TABLE_NAMES['clean_data_spark'], con=engine, if_exists="replace", index=False)
    spark.stop()


def spark_fe_1_func():
    engine = create_db_connection()
    conn = engine.connect()
    spark = SparkSession.builder.appName("Spark FE 1").getOrCreate()
    df = spark.read.jdbc(url=f"jdbc:postgresql://{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}", table=TABLE_NAMES['clean_data_spark'], properties={"user": PARAMS['db']['username'], "password": PARAMS['db']['password']})
    
    df = df.withColumn('trestbps_sqrt', sqrt(col('trestbps')))
    pandas_df = df.toPandas()
    pandas_df.to_sql(TABLE_NAMES['max_fe'], con=conn, if_exists="replace", index=False)
    spark.stop()

def spark_fe_2_func():
    engine = create_db_connection()
    conn = engine.connect()
    spark = SparkSession.builder.appName("Spark FE 2").getOrCreate()
    df = spark.read.jdbc(url=f"jdbc:postgresql://{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}", table=TABLE_NAMES['clean_data_spark'], properties={"user": PARAMS['db']['username'], "password": PARAMS['db']['password']})
    
    df = df.withColumn('age_trestbps', col('age') * col('trestbps'))
    pandas_df = df.toPandas()
    pandas_df.to_sql(TABLE_NAMES['product_fe'], con=conn, if_exists="replace", index=False)
    spark.stop()

def spark_model_training_lr_func():
    engine = create_db_connection()
    conn = engine.connect()
    spark = SparkSession.builder.appName("Spark Model Training LR").getOrCreate()
    df = spark.read.jdbc(url=f"jdbc:postgresql://{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}", table=TABLE_NAMES['max_fe'], properties={"user": PARAMS['db']['username'], "password": PARAMS['db']['password']})
    
    feature_columns = [col for col in df.columns if col != 'target']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df).select("features", "target")
    train_data, test_data = df.randomSplit([0.9, 0.1], seed=42)
    train_data = train_data.cache()
    test_data = test_data.cache()

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaler_model = scaler.fit(train_data)
    train_data = scaler_model.transform(train_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")
    test_data = scaler_model.transform(test_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")

    lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=10000)
    param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 1, 10, 100]).build()
    evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(train_data)
    best_model = cv_model.bestModel
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)

    results = pd.DataFrame({'model': ['Logistic Regression'], 'accuracy': [accuracy]})
    results.to_sql(TABLE_NAMES['model_results'], con=conn, if_exists="replace", index=False)
    spark.stop()

def spark_model_training_rf_func():
    engine = create_db_connection()
    conn = engine.connect()
    spark = SparkSession.builder.appName("Spark Model Training RF").getOrCreate()
    df = spark.read.jdbc(url=f"jdbc:postgresql://{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}", table=TABLE_NAMES['max_fe'], properties={"user": PARAMS['db']['username'], "password": PARAMS['db']['password']})
    
    feature_columns = [col for col in df.columns if col != 'target']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df).select("features", "target")
    train_data, test_data = df.randomSplit([0.9, 0.1], seed=42)
    train_data = train_data.cache()
    test_data = test_data.cache()

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaler_model = scaler.fit(train_data)
    train_data = scaler_model.transform(train_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")
    test_data = scaler_model.transform(test_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")

    rf = RandomForestClassifier(featuresCol="features", labelCol="target")
    param_grid = ParamGridBuilder().addGrid(rf.numTrees, [100, 200, 300]).addGrid(rf.maxDepth, [10, 20, 30]).build()
    evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(train_data)
    best_model = cv_model.bestModel
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)

    results = pd.DataFrame({'model': ['Random Forest'], 'accuracy': [accuracy]})
    results.to_sql(TABLE_NAMES['model_results'], con=conn, if_exists="replace", index=False)
    spark.stop()

def spark_model_training_gbt_func():
    engine = create_db_connection()
    conn = engine.connect()
    spark = SparkSession.builder.appName("Spark Model Training GBT").getOrCreate()
    df = spark.read.jdbc(url=f"jdbc:postgresql://{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}", table=TABLE_NAMES['max_fe'], properties={"user": PARAMS['db']['username'], "password": PARAMS['db']['password']})
    
    feature_columns = [col for col in df.columns if col != 'target']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df).select("features", "target")
    train_data, test_data = df.randomSplit([0.9, 0.1], seed=42)
    train_data = train_data.cache()
    test_data = test_data.cache()

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaler_model = scaler.fit(train_data)
    train_data = scaler_model.transform(train_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")
    test_data = scaler_model.transform(test_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")

    gbt = GBTClassifier(featuresCol="features", labelCol="target")
    param_grid = ParamGridBuilder().addGrid(gbt.maxIter, [100, 200, 300]).addGrid(gbt.maxDepth, [3, 5, 10]).addGrid(gbt.stepSize, [0.01, 0.1, 0.2]).build()
    evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(train_data)
    best_model = cv_model.bestModel
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)

    results = pd.DataFrame({'model': ['Gradient Boosting'], 'accuracy': [accuracy]})
    results.to_sql(TABLE_NAMES['model_results'], con=conn, if_exists="replace", index=False)
    spark.stop()

def evaluate_best_model():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['model_results']}", conn)
    
    best_model = df.loc[df['accuracy'].idxmax()]
    print(f"The best model is {best_model['model']} with accuracy {best_model['accuracy']}")
    
    conn.close()

# Define the DAG
dag = DAG(
    'yetayaltizale_hw4',
    default_args=default_args,
    description='Classify with feature engineering and model selection',
    schedule_interval=PARAMS['workflow']['workflow_schedule_interval'],
    tags=["de300"]
)

# Define tasks
add_data_to_table = PythonOperator(
    task_id='add_data_to_table',
    python_callable=add_data_to_table_func,
    dag=dag
)

eda = PythonOperator(
    task_id='eda',
    python_callable=eda_func,
    dag=dag
)

fe_1 = PythonOperator(
    task_id='fe_1',
    python_callable=fe_1_func,
    dag=dag
)

fe_2 = PythonOperator(
    task_id='fe_2',
    python_callable=fe_2_func,
    dag=dag
)

model_training_lr = PythonOperator(
    task_id='model_training_lr',
    python_callable=model_training_lr_func,
    dag=dag
)

model_training_rf = PythonOperator(
    task_id='model_training_rf',
    python_callable=model_training_rf_func,
    dag=dag
)

model_training_gbt = PythonOperator(
    task_id='model_training_gbt',
    python_callable=model_training_gbt_func,
    dag=dag
)

compare_models = PythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    dag=dag
)

evaluate_best_model = PythonOperator(
    task_id='evaluate_best_model',
    python_callable=evaluate_best_model,
    dag=dag
)

spark_clean_data = PythonOperator(
    task_id='spark_clean_data',
    python_callable=spark_clean_data_func,
    dag=dag
)

spark_fe_1 = PythonOperator(
    task_id='spark_fe_1',
    python_callable=spark_fe_1_func,
    dag=dag
)

spark_fe_2 = PythonOperator(
    task_id='spark_fe_2',
    python_callable=spark_fe_2_func,
    dag=dag
)

spark_model_training_lr = PythonOperator(
    task_id='spark_model_training_lr',
    python_callable=spark_model_training_lr_func,
    dag=dag
)

spark_model_training_rf = PythonOperator(
    task_id='spark_model_training_rf',
    python_callable=spark_model_training_rf_func,
    dag=dag
)

spark_model_training_gbt = PythonOperator(
    task_id='spark_model_training_gbt',
    python_callable=spark_model_training_gbt_func,
    dag=dag
)

# Define task dependencies
add_data_to_table >> [eda, spark_clean_data]

# Define dependencies for EDA and feature engineering tasks
eda >> fe_1
eda >> fe_2

# Define dependencies for model training tasks
fe_1 >> model_training_lr
fe_1 >> model_training_rf
fe_1 >> model_training_gbt

fe_2 >> model_training_lr
fe_2 >> model_training_rf
fe_2 >> model_training_gbt

# Define dependencies for Spark feature engineering and model training tasks
spark_clean_data >> spark_fe_1
spark_clean_data >> spark_fe_2

spark_fe_1 >> spark_model_training_lr
spark_fe_1 >> spark_model_training_rf
spark_fe_1 >> spark_model_training_gbt

spark_fe_2 >> spark_model_training_lr
spark_fe_2 >> spark_model_training_rf
spark_fe_2 >> spark_model_training_gbt

# Define dependencies for model comparison and evaluation
[model_training_lr, model_training_rf, model_training_gbt, spark_model_training_lr, spark_model_training_rf, spark_model_training_gbt] >> compare_models
compare_models >> evaluate_best_model
