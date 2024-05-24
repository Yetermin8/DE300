#!/usr/bin/env python
# coding: utf-8

# In[7]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
import requests
from scrapy import Selector
import re


# In[8]:


# Initialize Spark session
spark = SparkSession.builder \
    .appName("HW3 Spark Heart Disease Cleaning") \
    .getOrCreate()

# Load the previously saved DataFrame
df = spark.read.csv("s3://de300spring2024/yetayal_tizale/HW3_Spark/heart_disease.csv", header=True, inferSchema=True)

# Cleaning steps
# Getting rid of the nonsensical data after row 900
df = df.limit(899)

# Retaining only the required columns
columns_to_keep = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs',
                   'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang',
                   'oldpeak', 'slope', 'target']
df = df.select(*columns_to_keep)

# Impute missing values for binary variables
binary_cols = ['painloc', 'painexer', 'exang']
for col_name in binary_cols:
    mode_value = df.groupby(col_name).count().orderBy('count', ascending=False).first()[0]
    df = df.fillna({col_name: mode_value})

# Replace values less than 100 in 'trestbps' and impute missing values with the median
median_trestbps = df.approxQuantile('trestbps', [0.5], 0.01)[0]
df = df.withColumn('trestbps', when(df['trestbps'] < 100, median_trestbps).otherwise(df['trestbps']))
df = df.na.fill({'trestbps': median_trestbps})

# Replace outliers in 'oldpeak' and impute missing values with the median
median_oldpeak = df.approxQuantile('oldpeak', [0.5], 0.01)[0]
df = df.withColumn('oldpeak', when((df['oldpeak'] < 0) | (df['oldpeak'] > 4), median_oldpeak).otherwise(df['oldpeak']))
df = df.na.fill({'oldpeak': median_oldpeak})

# Impute missing values for 'thaldur' and 'thalach' with median
for col_name in ['thaldur', 'thalach']:
    median_value = df.approxQuantile(col_name, [0.5], 0.01)[0]
    df = df.na.fill({col_name: median_value})

# Replace invalid values in 'fbs', 'prop', 'nitr', 'pro', 'diuretic' and impute missing values with 0
binary_cols = ['fbs', 'prop', 'nitr', 'pro', 'diuretic']
for col_name in binary_cols:
    df = df.withColumn(col_name, when(df[col_name] > 1, 1).otherwise(df[col_name]))
    df = df.na.fill({col_name: 0})

# Impute missing values in 'slope' with the mode
mode_slope = df.groupby('slope').count().orderBy('count', ascending=False).first()[0]
df = df.na.fill({'slope': mode_slope})

# Filter rows where 'age' is not NaN and less than 18, or 75 and over
df = df.withColumn('age', col('age').cast(FloatType()))
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

# Save the cleaned DataFrame to a CSV file
df.write.format("csv").option("header", "true").mode("overwrite").save("s3://de300spring2024/yetayal_tizale/HW3_Spark/data/heart_disease_cleaned.csv")

spark.stop()


# In[ ]:




