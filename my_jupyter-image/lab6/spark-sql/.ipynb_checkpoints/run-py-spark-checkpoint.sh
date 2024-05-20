#!/bin/bash

# Remove any previous output files
/bin/rm -r -f ../data/transformed_data.csv

# Run the Spark job using spark-submit
/opt/spark/bin/spark-submit spark_sql_processing.py


