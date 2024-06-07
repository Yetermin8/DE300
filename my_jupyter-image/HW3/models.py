#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Model 1

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Model 1 Prediction") \
    .getOrCreate()

# Load data
data = spark.read.csv("s3://de300spring2024/yetayal_tizale/HW3_Spark/data/heart_disease_cleaned.csv/part-00000-6b7d28dc-33d4-4e1a-9ea0-4561c767fc45-c000.csv", header=True, inferSchema=True)

# Prepare the data
feature_columns = [col for col in data.columns if col != 'target']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "target")

# Split the data into training and test sets with stratification
train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)
train_data = train_data.cache()
test_data = test_data.cache()

# Standardize the data
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")
test_data = scaler_model.transform(test_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")

# Define models to train
lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=10000)
rf = RandomForestClassifier(featuresCol="features", labelCol="target")
dt = DecisionTreeClassifier(featuresCol="features", labelCol="target")

# Grid search parameters for tuning
param_grids = {
    'Logistic Regression': ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 1, 10, 100]) \
        .build(),
    'Random Forest': ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [10, 20, 30]) \
        .build(),
    'Decision Tree': ParamGridBuilder() \
        .addGrid(dt.maxDepth, [10, 20, 30]) \
        .addGrid(dt.minInstancesPerNode, [2, 5, 10]) \
        .build()
}

# Train models and tune hyperparameters
results = {}
evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")

for model_name, model in zip(['Logistic Regression', 'Random Forest', 'Decision Tree'], [lr, rf, dt]):
    print(f"Training {model_name}...")
    cv = CrossValidator(estimator=model, estimatorParamMaps=param_grids[model_name], evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(train_data)
    results[model_name] = {
        'best_model': cv_model.bestModel,
        'cv_results': cv_model.avgMetrics,
        'best_params': cv_model.bestModel.extractParamMap()
    }
    print(f"Best parameters for {model_name}: {results[model_name]['best_params']}")

# Evaluate on test data and find the best model
best_model_name = max(results, key=lambda x: max(results[x]['cv_results']))
best_model = results[best_model_name]['best_model']

predictions = best_model.transform(test_data)
accuracy = evaluator.evaluate(predictions)
print(f"\nBest Model: {best_model_name}")
print(f"Test Set Accuracy: {accuracy:.3f}")

# Print confusion matrix and classification report
conf_matrix = predictions.groupBy("target", "prediction").count().collect()
class_report = evaluator.evaluate(predictions)

# Save results to a folder
result_folder = "/tmp/HW3/model_results/model_1"
os.makedirs(result_folder, exist_ok=True)

# Save accuracy
with open(os.path.join(result_folder, "accuracy.txt"), "w") as f:
    f.write(f"Test Set Accuracy: {accuracy:.3f}\n")

# Save confusion matrix
with open(os.path.join(result_folder, "confusion_matrix.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    for row in conf_matrix:
        f.write(f"{row}\n")

# Save classification report
with open(os.path.join(result_folder, "classification_report.txt"), "w") as f:
    f.write(f"Classification Report:\n{class_report}\n")

# Save best parameters
with open(os.path.join(result_folder, "best_params.txt"), "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Best Model Parameters: {results[best_model_name]['best_params']}\n")
    f.write(f"Best Cross-Validation Score: {max(results[best_model_name]['cv_results']):.3f}\n")

print(f"Results saved to {result_folder}")

# Upload results to S3
os.system(f"aws s3 cp {result_folder} s3://de300spring2024/yetayal_tizale/HW3_Spark/model_results_emr/model_1 --recursive")

spark.stop()


# In[2]:


# Model 2

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Function to train, evaluate and save results for a model
def train_and_evaluate_model(model_name, model, param_grid, train_data, test_data, result_folder):
    evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(train_data)
    
    best_model = cv_model.bestModel
    best_params = best_model.extractParamMap()
    cv_results = cv_model.avgMetrics
    
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    
    conf_matrix = predictions.groupBy("target", "prediction").count().collect()
    class_report = evaluator.evaluate(predictions)
    
    os.makedirs(result_folder, exist_ok=True)
    
    with open(os.path.join(result_folder, "accuracy.txt"), "w") as f:
        f.write(f"Test Set Accuracy: {accuracy:.3f}\n")
    
    with open(os.path.join(result_folder, "confusion_matrix.txt"), "w") as f:
        f.write("Confusion Matrix:\n")
        for row in conf_matrix:
            f.write(f"{row}\n")
    
    with open(os.path.join(result_folder, "classification_report.txt"), "w") as f:
        f.write(f"Classification Report:\n{class_report}\n")
    
    with open(os.path.join(result_folder, "best_params.txt"), "w") as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"Best Model Parameters: {best_params}\n")
        f.write(f"Best Cross-Validation Score: {max(cv_results):.3f}\n")
    
    print(f"Results for {model_name} saved to {result_folder}")

# Initialize Spark session with increased stack size
spark = SparkSession.builder \
    .appName("Model 2 Prediction") \
    .config("spark.driver.extraJavaOptions", "-Xss4m") \
    .config("spark.executor.extraJavaOptions", "-Xss4m") \
    .getOrCreate()

# Load data
data = spark.read.csv("s3://de300spring2024/yetayal_tizale/HW3_Spark/data/heart_disease_cleaned.csv/part-00000-6b7d28dc-33d4-4e1a-9ea0-4561c767fc45-c000.csv", header=True, inferSchema=True)

# Create interaction terms
data = data.withColumn("age_trestbps", col("age") * col("trestbps"))
data = data.withColumn("smoke_oldpeak", col("smoke") * col("oldpeak"))

# Prepare the data
feature_columns = [col for col in data.columns if col != 'target']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "target")

# Split the data into training and test sets with stratification
train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)
train_data = train_data.cache()
test_data = test_data.cache()

# Standardize the data
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")
test_data = scaler_model.transform(test_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")

# Define the Gradient Boosting model
gbt = GBTClassifier(featuresCol="features", labelCol="target")

# Grid search parameters for tuning
param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [100, 200]) \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .addGrid(gbt.stepSize, [0.01, 0.1]) \
    .build()

result_folder = "/tmp/HW3/model_results/model_2/Gradient_Boosting"
train_and_evaluate_model('Gradient Boosting', gbt, param_grid, train_data, test_data, result_folder)

# Upload results to S3
os.system(f"aws s3 cp {result_folder} s3://de300spring2024/yetayal_tizale/HW3_Spark/model_results_emr/model_2 --recursive")

spark.stop()


# In[3]:


# Model 3

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p, sqrt
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Function to train, evaluate and save results for a model
def train_and_evaluate_model(model_name, model, param_grid, train_data, test_data, result_folder):
    evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(train_data)
    
    best_model = cv_model.bestModel
    best_params = best_model.extractParamMap()
    cv_results = cv_model.avgMetrics
    
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    
    conf_matrix = predictions.groupBy("target", "prediction").count().collect()
    class_report = evaluator.evaluate(predictions)
    
    os.makedirs(result_folder, exist_ok=True)
    
    with open(os.path.join(result_folder, "accuracy.txt"), "w") as f:
        f.write(f"Test Set Accuracy: {accuracy:.3f}\n")
    
    with open(os.path.join(result_folder, "confusion_matrix.txt"), "w") as f:
        f.write("Confusion Matrix:\n")
        for row in conf_matrix:
            f.write(f"{row}\n")
    
    with open(os.path.join(result_folder, "classification_report.txt"), "w") as f:
        f.write(f"Classification Report:\n{class_report}\n")
    
    with open(os.path.join(result_folder, "best_params.txt"), "w") as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"Best Model Parameters: {best_params}\n")
        f.write(f"Best Cross-Validation Score: {max(cv_results):.3f}\n")
    
    print(f"Results for {model_name} saved to {result_folder}")

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Model 3 Prediction") \
    .getOrCreate()

# Load data
data = spark.read.csv("s3://de300spring2024/yetayal_tizale/HW3_Spark/data/heart_disease_cleaned.csv/part-00000-6b7d28dc-33d4-4e1a-9ea0-4561c767fc45-c000.csv", header=True, inferSchema=True)

# Apply transformations
data = data.withColumn("sex", when(col("sex") == 0, 0).otherwise(1))
data = data.withColumn("smoke", log1p(col("smoke")))
data = data.withColumn("fbs", log1p(col("fbs")))
data = data.withColumn("prop", log1p(col("prop")))
data = data.withColumn("nitr", log1p(col("nitr")))
data = data.withColumn("pro", log1p(col("pro")))
data = data.withColumn("diuretic", log1p(col("diuretic")))
data = data.withColumn("oldpeak", sqrt(col("oldpeak")))
data = data.withColumn("cdc_smoke_rate", sqrt(col("cdc_smoke_rate")))

# Create interaction terms
data = data.withColumn("age_trestbps", col("age") * col("trestbps"))
data = data.withColumn("smoke_oldpeak", col("smoke") * col("oldpeak"))

# Prepare the data
feature_columns = [col for col in data.columns if col != 'target']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "target")

# Split the data into training and test sets with stratification
train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)
train_data = train_data.cache()
test_data = test_data.cache()

# Handling imbalanced data by adding a weight column
majority_class_weight = 1.0
minority_class_weight = train_data.filter(col('target') == 0).count() / train_data.filter(col('target') == 1).count()
train_data = train_data.withColumn('weight', when(col('target') == 1, minority_class_weight).otherwise(majority_class_weight))

# Standardize the data
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("scaledFeatures", "target", "weight").withColumnRenamed("scaledFeatures", "features")
test_data = scaler_model.transform(test_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")

# Setup parameter grid
lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=10000, weightCol="weight")
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.001, 0.01, 0.1, 1, 10, 100]) \
    .addGrid(lr.elasticNetParam, [0, 1]) \
    .build()

result_folder = "/tmp/HW3/model_results/model_3/Logistic_Regression"
train_and_evaluate_model('Logistic Regression', lr, param_grid, train_data, test_data, result_folder)

# Upload results to S3
os.system(f"aws s3 cp {result_folder} s3://de300spring2024/yetayal_tizale/HW3_Spark/model_results_emr/model_3 --recursive")

spark.stop()


# In[4]:


# Model 4

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Function to train, evaluate and save results for a model
def train_and_evaluate_model(model_name, model, param_grid, train_data, test_data, result_folder):
    evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(train_data)
    
    best_model = cv_model.bestModel
    best_params = best_model.extractParamMap()
    cv_results = cv_model.avgMetrics
    
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    
    conf_matrix = predictions.groupBy("target", "prediction").count().collect()
    class_report = evaluator.evaluate(predictions)
    
    os.makedirs(result_folder, exist_ok=True)
    
    with open(os.path.join(result_folder, "accuracy.txt"), "w") as f:
        f.write(f"Test Set Accuracy: {accuracy:.3f}\n")
    
    with open(os.path.join(result_folder, "confusion_matrix.txt"), "w") as f:
        f.write("Confusion Matrix:\n")
        for row in conf_matrix:
            f.write(f"{row}\n")
    
    with open(os.path.join(result_folder, "classification_report.txt"), "w") as f:
        f.write(f"Classification Report:\n{class_report}\n")
    
    with open(os.path.join(result_folder, "best_params.txt"), "w") as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"Best Model Parameters: {best_params}\n")
        f.write(f"Best Cross-Validation Score: {max(cv_results):.3f}\n")
    
    print(f"Results for {model_name} saved to {result_folder}")

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Model 4 Prediction") \
    .getOrCreate()

# Load data
data = spark.read.csv("s3://de300spring2024/yetayal_tizale/HW3_Spark/data/heart_disease_cleaned.csv/part-00000-6b7d28dc-33d4-4e1a-9ea0-4561c767fc45-c000.csv", header=True, inferSchema=True)

# Prepare the data
feature_columns = [col for col in data.columns if col != 'target']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "target")

# Split the data into training and test sets with stratification
train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)
train_data = train_data.cache()
test_data = test_data.cache()

# Standardize the data
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")
test_data = scaler_model.transform(test_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")

# Define models to train
lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=10000)
rf = RandomForestClassifier(featuresCol="features", labelCol="target")

# Grid search parameters for tuning
param_grids = {
    'Logistic Regression': ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 1, 10, 100]) \
        .addGrid(lr.elasticNetParam, [0, 1]) \
        .build(),
    'Random Forest': ParamGridBuilder() \
        .addGrid(rf.numTrees, [100, 200, 300]) \
        .addGrid(rf.maxDepth, [10, 20, 30]) \
        .addGrid(rf.minInstancesPerNode, [2, 5, 10]) \
        .build()
}

# Train models and tune hyperparameters
for model_name, model in zip(['Logistic Regression', 'Random Forest'], [lr, rf]):
    print(f"Training {model_name}...")
    result_folder = f"/tmp/HW3/model_results/model_4/{model_name.replace(' ', '_')}"
    train_and_evaluate_model(model_name, model, param_grids[model_name], train_data, test_data, result_folder)

# Upload results to S3
os.system(f"aws s3 cp {result_folder} s3://de300spring2024/yetayal_tizale/HW3_Spark/model_results_emr/model_4 --recursive")

spark.stop()


# In[5]:


# Model 5

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, sqrt
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Function to train, evaluate and save results for a model
def train_and_evaluate_model(model_name, model, param_grid, train_data, test_data, result_folder):
    evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(train_data)
    
    best_model = cv_model.bestModel
    best_params = best_model.extractParamMap()
    cv_results = cv_model.avgMetrics
    
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    
    conf_matrix = predictions.groupBy("target", "prediction").count().collect()
    class_report = evaluator.evaluate(predictions)
    
    os.makedirs(result_folder, exist_ok=True)
    
    with open(os.path.join(result_folder, "accuracy.txt"), "w") as f:
        f.write(f"Test Set Accuracy: {accuracy:.3f}\n")
    
    with open(os.path.join(result_folder, "confusion_matrix.txt"), "w") as f:
        f.write("Confusion Matrix:\n")
        for row in conf_matrix:
            f.write(f"{row}\n")
    
    with open(os.path.join(result_folder, "classification_report.txt"), "w") as f:
        f.write(f"Classification Report:\n{class_report}\n")
    
    with open(os.path.join(result_folder, "best_params.txt"), "w") as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"Best Model Parameters: {best_params}\n")
        f.write(f"Best Cross-Validation Score: {max(cv_results):.3f}\n")
    
    print(f"Results for {model_name} saved to {result_folder}")

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Model 5 Prediction") \
    .getOrCreate()

# Load data
data = spark.read.csv("s3://de300spring2024/yetayal_tizale/HW3_Spark/data/heart_disease_cleaned.csv/part-00000-6b7d28dc-33d4-4e1a-9ea0-4561c767fc45-c000.csv", header=True, inferSchema=True)

# Apply transformations to skewed variables
data = data.withColumn('log_smoke', log1p(col('smoke')))
data = data.withColumn('sqrt_oldpeak', sqrt(col('oldpeak')))

# Create interaction terms for highly correlated variables
data = data.withColumn('age_trestbps', col('age') * col('trestbps'))

# Prepare the data
feature_columns = [col for col in data.columns if col != 'target']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "target")

# Split the data into training and test sets with stratification
train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)
train_data = train_data.cache()
test_data = test_data.cache()

# Standardize the data
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")
test_data = scaler_model.transform(test_data).select("scaledFeatures", "target").withColumnRenamed("scaledFeatures", "features")

# Define models to train
lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=10000)
rf = RandomForestClassifier(featuresCol="features", labelCol="target")
gbt = GBTClassifier(featuresCol="features", labelCol="target")

# Grid search parameters for tuning
param_grids = {
    'Logistic Regression': ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 1, 10, 100]) \
        .addGrid(lr.elasticNetParam, [0, 1]) \
        .build(),
    'Random Forest': ParamGridBuilder() \
        .addGrid(rf.numTrees, [100, 200, 300]) \
        .addGrid(rf.maxDepth, [10, 20, 30]) \
        .addGrid(rf.minInstancesPerNode, [2, 5, 10]) \
        .build(),
    'Gradient Boosting': ParamGridBuilder() \
        .addGrid(gbt.maxIter, [100, 200, 300]) \
        .addGrid(gbt.maxDepth, [3, 5, 10]) \
        .addGrid(gbt.stepSize, [0.01, 0.1, 0.2]) \
        .build()
}

# Train models and tune hyperparameters
for model_name, model in zip(['Logistic Regression', 'Random Forest', 'Gradient Boosting'], [lr, rf, gbt]):
    print(f"Training {model_name}...")
    result_folder = f"/tmp/HW3/model_results/model_5/{model_name.replace(' ', '_')}"
    train_and_evaluate_model(model_name, model, param_grids[model_name], train_data, test_data, result_folder)

# Upload results to S3
os.system(f"aws s3 cp {result_folder} s3://de300spring2024/yetayal_tizale/HW3_Spark/model_results_emr/model_5 --recursive")

spark.stop()


# In[ ]:

