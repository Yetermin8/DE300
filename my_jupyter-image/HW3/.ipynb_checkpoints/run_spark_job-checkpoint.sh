docker run -v ~/DE300/MY_DE300/DE300/my_jupyter-image/HW3:/tmp/HW3 \
           -p 8888:8888 \
           --name spark-sql-container \
           pyspark-image
