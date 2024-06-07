#!/bin/bash

# Upgrade pip
python3 -m pip install --upgrade pip

# Upgrade to the systems
sudo apt-get update
sudo apt-get install --only-upgrade openssl

# Install packages
sudo python3 -m pip install boto3 pandas "s3fs<=0.4" numpy requests scrapy
