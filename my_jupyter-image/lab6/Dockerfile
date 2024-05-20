# Use the latest Apache Spark image with Python support
FROM apache/spark-py:latest

# Run as root to install additional software
USER root

# Set the working directory to /tmp for any subsequent commands
WORKDIR /tmp

# Update the package lists and upgrade the installed packages
RUN apt update -y && apt upgrade -y

# Install Python and necessary packages globally
RUN apt install -y python3.10 python3-pip
RUN pip3 install numpy jupyter pyspark==3.4.0

# Copy the lab6 directory to /tmp
COPY lab6 /tmp/lab6

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Make the start.sh script executable
RUN chmod +x /tmp/lab6/start.sh

# Set the entrypoint to the custom start.sh script
ENTRYPOINT ["/bin/bash", "/tmp/lab6/start.sh"]
