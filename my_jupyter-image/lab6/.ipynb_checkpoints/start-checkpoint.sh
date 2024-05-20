#!/bin/bash

# Navigate to the spark-sql directory
cd /tmp/lab6/spark-sql

# Execute the Spark job script
bash run-py-spark.sh

# Keep the container running
tail -f /dev/null
