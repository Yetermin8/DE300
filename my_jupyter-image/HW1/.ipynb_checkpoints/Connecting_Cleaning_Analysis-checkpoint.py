import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

# Establish a connection to the MySQL database
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Yetayalmysql101!",
    database="data_eng_300_hw1"
)

# Create a cursor object to execute SQL queries
cursor = connection.cursor()

# Execute a SQL query to fetch data
cursor.execute("SELECT * FROM heart_disease")

# Fetch all rows from the result set
rows = cursor.fetchall()

# Get column names
column_names = [desc[0] for desc in cursor.description]

# Create a DataFrame from the fetched data
df = pd.DataFrame(rows, columns=column_names)

# Close cursor and connection
cursor.close()
connection.close()


non_categorical_variables = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thaldur', 'thaltime',  'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'oldpeak', 'rldv5', 'rldv5e', 'ca', 'restckm',  'exerckm', 'restef', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr']


import os

# Create a directory to save box plots
os.makedirs("box_plots", exist_ok=True)

# Iterate over each non-categorical variable
for var in non_categorical_variables:
    # Create a box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(df[var])
    plt.title(f"Box plot of {var}")
    plt.xlabel(var)
    plt.ylabel("Values")
    
    # Identify outliers
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Remove outliers
    df_cleaned = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)]
    
    # Save box plot
    plt.savefig(f"box_plots/{var}_boxplot.png")
    
    # Close the plot to avoid displaying multiple plots
    plt.close()

# Display the saved box plots
box_plot_files = os.listdir("box_plots")
print("Box plots saved successfully:")
print(box_plot_files)
