from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import boto3
import tomli
from io import StringIO
from sqlalchemy import create_engine
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sqrt
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from datetime import timedelta
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
import requests
from scrapy import Selector
import re
import logging
import botocore


# Define default_args for the DAG
default_args = {
    'owner': 'yetayaltizale',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': False,
    'email_on_retry': False,
}

# Define the DAG
dag = DAG(
    'yetayaltizale_hw4',
    default_args=default_args,
    description='Classify with feature engineering and model selection',
    schedule_interval=timedelta(days=1),
    tags=["de300"]
)

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
    "combined_fe": "heart_disease_combined_fe_features",
    "eda_results": "heart_disease_eda_results",
    "model_results": "heart_disease_model_results"
}

def read_config_from_s3() -> dict:
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=CONFIG_BUCKET, Key=CONFIG_FILE_KEY)
    config_data = obj['Body'].read().decode('utf-8')
    params = tomli.loads(config_data)
    return params

PARAMS = read_config_from_s3()

def create_db_connection():
    conn_uri = f"{PARAMS['db']['db_alchemy_driver']}://{PARAMS['db']['username']}:{PARAMS['db']['password']}@{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}"
    engine = create_engine(conn_uri)
    return engine

def add_data_to_table_func():
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

def clean_and_impute_data(df):
    # Retain only the required columns
    columns_to_keep = [
        'age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs',
        'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang',
        'oldpeak', 'slope', 'target'
    ]
    df = df[columns_to_keep]

    # a. Binary variables (painloc, painexer, exang)
    for col in ['painloc', 'painexer', 'exang']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # b. 'trestbps': Replace values less than 100 mm Hg with the median
    df['trestbps'] = np.where(df['trestbps'] < 100, df['trestbps'].median(), df['trestbps'])
    df['trestbps'].fillna(df['trestbps'].median(), inplace=True)

    # c. 'oldpeak': Replace values less than 0 and those greater than 4 with the median
    df['oldpeak'] = np.where((df['oldpeak'] < 0) | (df['oldpeak'] > 4), df['oldpeak'].median(), df['oldpeak'])
    df['oldpeak'].fillna(df['oldpeak'].median(), inplace=True)

    # d. 'thaldur' (duration of exercise), 'thalach' (max heart rate)
    for col in ['thaldur', 'thalach']:
        df[col].fillna(df[col].median(), inplace=True)

    # e. Replace missing values and values greater than 1 in 'fbs', 'prop', 'nitr', 'pro', 'diuretic'
    for col in ['fbs', 'prop', 'nitr', 'pro', 'diuretic']:
        df[col] = np.where(df[col] > 1, 1, df[col])
        df[col].fillna(0, inplace=True)

    # f. 'slope': Replace missing values with the mode
    df['slope'].fillna(df['slope'].mode()[0], inplace=True)

    # Convert 'age' to numeric, setting errors to NaN and filter out ages not in a reasonable range
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[(df['age'] >= 18) & (df['age'] < 75)]

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
            return abs_rates.get(age_group, {}).get(sex_key, np.nan)
        elif dataset == 'cdc':
            age_group = map_age_to_cdc_group(age)
            sex_key = 'male_rate' if sex == 1 else 'female_rate'
            return cdc_rates.get(age_group, {}).get(sex_key, np.nan)
        return np.nan

    # Create new columns for ABS and CDC rates
    df['abs_smoke_rate'] = df.apply(lambda row: get_smoking_rate(row['age'], row['sex'], 'abs'), axis=1)
    df['cdc_smoke_rate'] = df.apply(lambda row: get_smoking_rate(row['age'], row['sex'], 'cdc'), axis=1)

    # Function to get the average rate
    def get_average_smoking_rate(age, sex):
        abs_age_group = map_age_to_abs_group(age)
        cdc_age_group = map_age_to_cdc_group(age)
        abs_sex_key = 'Males' if sex == 1 else 'Females'
        cdc_sex_key = 'male_rate' if sex == 1 else 'female_rate'
        
        abs_rate = abs_rates.get(abs_age_group, {}).get(abs_sex_key, np.nan)
        cdc_rate = cdc_rates.get(cdc_age_group, {}).get(cdc_sex_key, np.nan)
        
        rates = [r for r in [abs_rate, cdc_rate] if not np.isnan(r)]
        return np.mean(rates) if rates else np.nan

    # Impute missing 'smoke' values using the mean of ABS and CDC rates
    def impute_smoking_values(row):
        if pd.isnull(row['smoke']):
            return get_average_smoking_rate(row['age'], row['sex'])
        return row['smoke']

    # Apply the imputation across the DataFrame
    df['smoke'] = df.apply(impute_smoking_values, axis=1)

    # Apply rounding to the specific columns
    df[['smoke', 'abs_smoke_rate', 'cdc_smoke_rate']] = df[['smoke', 'abs_smoke_rate', 'cdc_smoke_rate']].round(3)

    return df

def cleaning_and_eda_func():
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

        # Convert non-numeric values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Clean and impute data
        cleaned_df = clean_and_impute_data(df)

        # Save cleaned data to SQL table
        cleaned_df.to_sql(TABLE_NAMES['clean_data_pandas'], con=conn, if_exists="replace", index=False)

        # Correlation Analysis
        corr_matrix = cleaned_df.corr()
        corr_flat = corr_matrix.unstack().reset_index()
        corr_flat.columns = ['Feature1', 'Feature2', 'Correlation']
        corr_flat = corr_flat[corr_flat['Feature1'] != corr_flat['Feature2']]
        corr_flat['Abs_Correlation'] = corr_flat['Correlation'].abs()
        sorted_corr = corr_flat.sort_values(by='Abs_Correlation', ascending=False).head(10)
        sorted_corr.to_sql(TABLE_NAMES['eda_results'], con=conn, if_exists="replace", index=False)
    except Exception as e:
        print(f"Error in cleaning_and_eda function: {e}")
        raise
    finally:
        conn.close()

def fe_1_func():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['clean_data_pandas']}", conn)
    df['age_trestbps'] = df['age'] * df['trestbps']
    df.to_sql(TABLE_NAMES['max_fe'], con=conn, if_exists="replace", index=False)
    conn.close()

def fe_2_func():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['clean_data_pandas']}", conn)
    df['trestbps_squared'] = df['trestbps'] ** 2
    df.to_sql(TABLE_NAMES['product_fe'], con=conn, if_exists="replace", index=False)
    conn.close()

def fe_combined_func():
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['clean_data_pandas']}", conn)
    df['age_trestbps'] = df['age'] * df['trestbps']
    df['trestbps_squared'] = df['trestbps'] ** 2
    df.to_sql(TABLE_NAMES['combined_fe'], con=conn, if_exists="replace", index=False)
    conn.close()

def model_training_func(model_name, fe_table_name, **kwargs):
    ti = kwargs['ti']
    engine = create_db_connection()
    conn = engine.connect()
    df = pd.read_sql(f"SELECT * FROM {fe_table_name}", conn)
    X = df.drop('target', axis=1)
    y = df['target']
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Push the accuracy to XCom
    ti.xcom_push(key=f'{model_name}_accuracy', value=accuracy)
    ti.xcom_push(key=f'{model_name}_fe_table', value=fe_table_name)
    conn.close()

def compare_models_func(**kwargs):
    ti = kwargs['ti']
    model_accuracies = {
        'Logistic Regression': ti.xcom_pull(key='Logistic Regression_accuracy', task_ids=['model_training_lr_fe_1', 'model_training_lr_fe_2', 'model_training_lr_fe_combined']),
        'Random Forest': ti.xcom_pull(key='Random Forest_accuracy', task_ids=['model_training_rf_fe_1', 'model_training_rf_fe_2', 'model_training_rf_fe_combined']),
        'Gradient Boosting': ti.xcom_pull(key='Gradient Boosting_accuracy', task_ids=['model_training_gbt_fe_1', 'model_training_gbt_fe_2', 'model_training_gbt_fe_combined']),
    }

    best_model = max(model_accuracies, key=lambda k: max(model_accuracies[k]))
    best_accuracy = max(model_accuracies[best_model])
    
    print(f"The best model is {best_model} with accuracy {best_accuracy}")

def create_spark_session():
    return SparkSession.builder \
        .appName("Heart Disease Analysis HW4") \
        .getOrCreate()

def spark_clean_and_eda_func(input_path, output_path, **kwargs):
    spark = create_spark_session()
    try:
        df = spark.read.csv(input_path, header=True, inferSchema=True)

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
            if (response.status_code == 200):
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

        df.write.csv(output_path, mode='overwrite', header=True)
        return None
    finally:
        spark.stop()

def list_s3_objects(bucket_name, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response.get('Contents', []):
        logging.info(f"Found S3 object: {obj['Key']}")

def load_spark_cleaned_data(**kwargs):
    try:
        logging.info("Starting to load cleaned data from S3")
        
        s3 = boto3.client('s3')
        bucket_name = 'de300spring2024-yetayaltizale-airflow'
        key = 'HW4/heart_disease_cleaned.csv'
        local_path = '/tmp/heart_disease_cleaned.csv'

        # List S3 objects to verify the path and file
        list_s3_objects(bucket_name, 'HW4/')

        # Check if the object exists
        s3.head_object(Bucket=bucket_name, Key=key)
        
        # Download the object
        s3.download_file(bucket_name, key, local_path)
        logging.info(f"Downloaded {key} from bucket {bucket_name} to {local_path}")

        spark = create_spark_session()
        df = spark.read.csv(local_path, header=True, inferSchema=True)

        logging.info(f"Data count after loading: {df.count()}")
        df.show(5)
        
        # Save to Spark's local directory for easier access in subsequent tasks
        spark_save_path = '/tmp/heart_disease_cleaned_spark.csv'
        df.write.csv(spark_save_path, mode='overwrite', header=True)

        # Push the path to XCom for the next task to use
        ti = kwargs['ti']
        ti.xcom_push(key='spark_cleaned_data_path', value=spark_save_path)

        return spark_save_path
    
    except boto3.exceptions.S3UploadFailedError as e:
        logging.error(f"S3 upload failed: {e}")
        raise
    except boto3.exceptions.S3DownloadError as e:
        logging.error(f"S3 download failed: {e}")
        raise
    except botocore.exceptions.ClientError as e:
        logging.error(f"Client error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading cleaned data from S3: {e}")
        raise


def spark_fe_1_func(input_path, output_path, **kwargs):
    spark = create_spark_session()
    try:
        logging.info(f"Reading CSV file from {input_path}")
        df = spark.read.csv(input_path, header=True, inferSchema=True)
        
        # Log the schema and columns
        logging.info("DataFrame Schema: ")
        df.printSchema()
        logging.info("DataFrame Columns: ")
        logging.info(df.columns)
        
        if 'trestbps' not in df.columns:
            logging.error("Column 'trestbps' not found in DataFrame")
            raise ValueError("Column 'trestbps' not found in DataFrame")

        df = df.withColumn('trestbps_sqrt', sqrt(col('trestbps')))
        logging.info(f"Writing processed data to {output_path}")
        df.write.csv(output_path, mode='overwrite', header=True)
        logging.info(f"Data successfully written to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error in spark_fe_1_func: {e}")
        raise
    finally:
        spark.stop()



def spark_fe_2_func(input_path, output_path, **kwargs):
    spark = create_spark_session()
    try:
        logging.info(f"Reading CSV file from {input_path}")
        df = spark.read.csv(input_path, header=True, inferSchema=True)
        df = df.withColumn('age_trestbps', col('age') * col('trestbps'))
        logging.info(f"Writing processed data to {output_path}")
        df.write.csv(output_path, mode='overwrite', header=True)
        logging.info(f"Data successfully written to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error in spark_fe_2_func: {e}")
        raise
    finally:
        spark.stop()

def spark_fe_combined_func(input_path, output_path, **kwargs):
    spark = create_spark_session()
    try:
        logging.info(f"Reading CSV file from {input_path}")
        df = spark.read.csv(input_path, header=True, inferSchema=True)
        df = df.withColumn('trestbps_sqrt', sqrt(col('trestbps')))
        df = df.withColumn('age_trestbps', col('age') * col('trestbps'))
        logging.info(f"Writing processed data to {output_path}")
        df.write.csv(output_path, mode='overwrite', header=True)
        logging.info(f"Data successfully written to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error in spark_fe_combined_func: {e}")
        raise
    finally:
        spark.stop()

def spark_model_training_func(input_path, fe_table_name, **kwargs):
    ti = kwargs['ti']
    spark = create_spark_session()
    try:
        df = spark.read.parquet(input_path)
        feature_columns = [col for col in df.columns if col != 'target']

        # Define the assembler
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

        # Logistic Regression
        lr = LogisticRegression(labelCol="target", featuresCol="features", maxIter=10000)
        param_grid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 1, 10, 100]).build()
        evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
        pipeline_lr = Pipeline(stages=[assembler, lr])
        cv_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=param_grid_lr, evaluator=evaluator, numFolds=5)

        # Random Forest
        rf = RandomForestClassifier(labelCol="target", featuresCol="features")
        param_grid_rf = ParamGridBuilder().addGrid(rf.numTrees, [100, 200, 300]).addGrid(rf.maxDepth, [10, 20, 30]).build()
        pipeline_rf = Pipeline(stages=[assembler, rf])
        cv_rf = CrossValidator(estimator=pipeline_rf, estimatorParamMaps=param_grid_rf, evaluator=evaluator, numFolds=5)

        # Gradient Boosting
        gbt = GBTClassifier(labelCol="target", featuresCol="features")
        param_grid_gbt = ParamGridBuilder().addGrid(gbt.maxIter, [100, 200, 300]).addGrid(gbt.maxDepth, [3, 5, 10]).addGrid(gbt.stepSize, [0.01, 0.1, 0.2]).build()
        pipeline_gbt = Pipeline(stages=[assembler, gbt])
        cv_gbt = CrossValidator(estimator=pipeline_gbt, estimatorParamMaps=param_grid_gbt, evaluator=evaluator, numFolds=5)

        # Split data into training and test sets
        train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

        # Train and evaluate Logistic Regression model
        cv_model_lr = cv_lr.fit(train_df)
        best_model_lr = cv_model_lr.bestModel
        predictions_lr = best_model_lr.transform(test_df)
        accuracy_lr = evaluator.evaluate(predictions_lr)

        # Train and evaluate Random Forest model
        cv_model_rf = cv_rf.fit(train_df)
        best_model_rf = cv_model_rf.bestModel
        predictions_rf = best_model_rf.transform(test_df)
        accuracy_rf = evaluator.evaluate(predictions_rf)

        # Train and evaluate Gradient Boosting model
        cv_model_gbt = cv_gbt.fit(train_df)
        best_model_gbt = cv_model_gbt.bestModel
        predictions_gbt = best_model_gbt.transform(test_df)
        accuracy_gbt = evaluator.evaluate(predictions_gbt)

        # Push the accuracies to XCom
        ti.xcom_push(key='Logistic Regression_accuracy', value=accuracy_lr)
        ti.xcom_push(key='Random Forest_accuracy', value=accuracy_rf)
        ti.xcom_push(key='Gradient Boosting_accuracy', value=accuracy_gbt)
        ti.xcom_push(key='Logistic Regression_fe_table', value=fe_table_name)
        ti.xcom_push(key='Random Forest_fe_table', value=fe_table_name)
        ti.xcom_push(key='Gradient Boosting_fe_table', value=fe_table_name)
    finally:
        spark.stop()

def evaluate_best_model_func(**kwargs):
    ti = kwargs['ti']
    model_accuracies = {
        'Logistic Regression': ti.xcom_pull(key='Logistic Regression_accuracy', task_ids=['spark_model_training_lr_rf_gbt_fe_1', 'spark_model_training_lr_rf_gbt_fe_2', 'spark_model_training_lr_rf_gbt_fe_combined']),
        'Random Forest': ti.xcom_pull(key='Random Forest_accuracy', task_ids=['spark_model_training_lr_rf_gbt_fe_1', 'spark_model_training_lr_rf_gbt_fe_2', 'spark_model_training_lr_rf_gbt_fe_combined']),
        'Gradient Boosting': ti.xcom_pull(key='Gradient Boosting_accuracy', task_ids=['spark_model_training_lr_rf_gbt_fe_1', 'spark_model_training_lr_rf_gbt_fe_2', 'spark_model_training_lr_rf_gbt_fe_combined']),
    }

    best_model = max(model_accuracies, key=lambda k: max(model_accuracies[k]))
    best_accuracy = max(model_accuracies[best_model])
    
    print(f"The best model is {best_model} with accuracy {best_accuracy}")

# Define Airflow tasks
add_data_to_table = PythonOperator(
    task_id='add_data_to_table',
    python_callable=add_data_to_table_func,
    dag=dag
)

cleaning_and_eda = PythonOperator(
    task_id='cleaning_and_eda',
    python_callable=cleaning_and_eda_func,
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

fe_combined = PythonOperator(
    task_id='fe_combined',
    python_callable=fe_combined_func,
    dag=dag
)

model_training_lr_fe_1 = PythonOperator(
    task_id='model_training_lr_fe_1',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Logistic Regression', 'fe_table_name': TABLE_NAMES['max_fe']},
    dag=dag
)

model_training_rf_fe_1 = PythonOperator(
    task_id='model_training_rf_fe_1',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Random Forest', 'fe_table_name': TABLE_NAMES['max_fe']},
    dag=dag
)

model_training_gbt_fe_1 = PythonOperator(
    task_id='model_training_gbt_fe_1',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Gradient Boosting', 'fe_table_name': TABLE_NAMES['max_fe']},
    dag=dag
)

model_training_lr_fe_2 = PythonOperator(
    task_id='model_training_lr_fe_2',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Logistic Regression', 'fe_table_name': TABLE_NAMES['product_fe']},
    dag=dag
)

model_training_rf_fe_2 = PythonOperator(
    task_id='model_training_rf_fe_2',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Random Forest', 'fe_table_name': TABLE_NAMES['product_fe']},
    dag=dag
)

model_training_gbt_fe_2 = PythonOperator(
    task_id='model_training_gbt_fe_2',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Gradient Boosting', 'fe_table_name': TABLE_NAMES['product_fe']},
    dag=dag
)

model_training_lr_fe_combined = PythonOperator(
    task_id='model_training_lr_fe_combined',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Logistic Regression', 'fe_table_name': TABLE_NAMES['combined_fe']},
    dag=dag
)

model_training_rf_fe_combined = PythonOperator(
    task_id='model_training_rf_fe_combined',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Random Forest', 'fe_table_name': TABLE_NAMES['combined_fe']},
    dag=dag
)

model_training_gbt_fe_combined = PythonOperator(
    task_id='model_training_gbt_fe_combined',
    python_callable=model_training_func,
    op_kwargs={'model_name': 'Gradient Boosting', 'fe_table_name': TABLE_NAMES['combined_fe']},
    dag=dag
)

compare_models_task = PythonOperator(
    task_id='compare_models',
    python_callable=compare_models_func,
    dag=dag
)

evaluate_best_model_task = PythonOperator(
    task_id='evaluate_best_model',
    python_callable=evaluate_best_model_func,
    dag=dag
)

spark_clean_and_eda_func_task = PythonOperator(
    task_id='spark_clean_and_eda',
    python_callable=spark_clean_and_eda_func,
    op_kwargs={'input_path': '/tmp/heart_disease.csv', 'output_path': '/tmp/heart_disease_cleaned_spark.parquet'},
    dag=dag,
)

load_spark_cleaned_data_task = PythonOperator(
    task_id='load_spark_cleaned_data',
    python_callable=load_spark_cleaned_data,
    dag=dag,
)

spark_fe_1 = PythonOperator(
    task_id='spark_fe_1',
    python_callable=spark_fe_1_func,
    op_kwargs={
        'input_path': '{{ ti.xcom_pull(task_ids="load_spark_cleaned_data") }}',
        'output_path': '/tmp/spark_fe_1_heart_disease.csv'
    },
    dag=dag
)

spark_fe_2 = PythonOperator(
    task_id='spark_fe_2',
    python_callable=spark_fe_2_func,
    op_kwargs={
        'input_path': '/tmp/spark_fe_1_heart_disease.csv',
        'output_path': '/tmp/spark_fe_2_heart_disease.csv'
    },
    dag=dag
)

spark_fe_combined = PythonOperator(
    task_id='spark_fe_combined',
    python_callable=spark_fe_combined_func,
    op_kwargs={
        'input_path': '/tmp/spark_fe_2_heart_disease.csv',
        'output_path': '/tmp/spark_combined_fe_heart_disease.csv'
    },
    dag=dag
)

spark_model_training_lr_rf_gbt_fe_1 = PythonOperator(
    task_id='spark_model_training_lr_rf_gbt_fe_1',
    python_callable=spark_model_training_func,
    op_kwargs={'input_path': '/tmp/spark_fe_1_heart_disease.csv', 'fe_table_name': 'spark_fe_1'},
    dag=dag,
    provide_context=True  # Ensure context is provided for XCom pushing
)

spark_model_training_lr_rf_gbt_fe_2 = PythonOperator(
    task_id='spark_model_training_lr_rf_gbt_fe_2',
    python_callable=spark_model_training_func,
    op_kwargs={'input_path': '/tmp/spark_fe_2_heart_disease.csv', 'fe_table_name': 'spark_fe_2'},
    dag=dag,
    provide_context=True  # Ensure context is provided for XCom pushing
)

spark_model_training_lr_rf_gbt_fe_combined = PythonOperator(
    task_id='spark_model_training_lr_rf_gbt_fe_combined',
    python_callable=spark_model_training_func,
    op_kwargs={'input_path': '/tmp/spark_combined_fe_heart_disease.csv', 'fe_table_name': 'spark_fe_combined'},
    dag=dag,
    provide_context=True  # Ensure context is provided for XCom pushing
)


# Define task dependencies
add_data_to_table >> cleaning_and_eda
add_data_to_table >> spark_clean_and_eda_func_task

cleaning_and_eda >> fe_1
cleaning_and_eda >> fe_2
cleaning_and_eda >> fe_combined

fe_1 >> model_training_lr_fe_1
fe_1 >> model_training_rf_fe_1
fe_1 >> model_training_gbt_fe_1

fe_2 >> model_training_lr_fe_2
fe_2 >> model_training_rf_fe_2
fe_2 >> model_training_gbt_fe_2

fe_combined >> model_training_lr_fe_combined
fe_combined >> model_training_rf_fe_combined
fe_combined >> model_training_gbt_fe_combined

spark_clean_and_eda_func_task >> load_spark_cleaned_data_task
load_spark_cleaned_data_task >> spark_fe_1
spark_fe_1 >> spark_fe_2
spark_fe_2 >> spark_fe_combined

spark_fe_1 >> spark_model_training_lr_rf_gbt_fe_1
spark_fe_2 >> spark_model_training_lr_rf_gbt_fe_2
spark_fe_combined >> spark_model_training_lr_rf_gbt_fe_combined

[model_training_lr_fe_1, model_training_rf_fe_1, model_training_gbt_fe_1,
 model_training_lr_fe_2, model_training_rf_fe_2, model_training_gbt_fe_2,
 model_training_lr_fe_combined, model_training_rf_fe_combined, model_training_gbt_fe_combined,
 spark_model_training_lr_rf_gbt_fe_1, spark_model_training_lr_rf_gbt_fe_2, spark_model_training_lr_rf_gbt_fe_combined] >> compare_models_task

compare_models_task >> evaluate_best_model_task
