# Use the latest Apache Spark image with Python support
FROM apache/spark-py:latest

# Run as root to install additional software
USER root

# Set the working directory to /tmp for any subsequent commands
WORKDIR /tmp

# Update the package lists and upgrade the installed packages
RUN apt update -y && apt upgrade -y

# Install python3.10 virtual environment tools and pip
RUN apt install -y python3.10-venv python3-pip

# Create a virtual environment for the project
RUN python3 -m venv demos

# Activate the virtual environment and install necessary Python packages
RUN /bin/bash -c "source demos/bin/activate && \
    pip install numpy venv-pack jupyter && \
    pip install pyspark==3.4.0 && \
    pip install scrapy && \
    demos/bin/venv-pack -o demos.tar.gz"

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Add a command to start Jupyter Notebook when the container starts
CMD ["/bin/bash", "-c", "source demos/bin/activate && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"]
