import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sqlalchemy import create_engine

try:
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

finally:
    # Close cursor and connection
    cursor.close()
    connection.close()

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

# Plotting box plots
os.makedirs("box_plots", exist_ok=True)
plt.figure(figsize=(16, 12))
for i, var in enumerate(non_categorical_variables, 1):
    plt.subplot((len(non_categorical_variables) + 3) // 4, 4, i)
    sns.boxplot(y=df[var])
    plt.title(f"{var}")
plt.tight_layout()
plt.savefig("box_plots/all_variables_boxplot.png")
plt.close()

# Correlation Analysis
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlation Matrix')
plt.savefig('correlation_heatmap.png')
plt.close()

# Top Correlated Pairs
corr_flat = corr_matrix.unstack().reset_index()
corr_flat.columns = ['Feature1', 'Feature2', 'Correlation']
corr_flat = corr_flat[corr_flat['Feature1'] != corr_flat['Feature2']]
corr_flat['Abs_Correlation'] = corr_flat['Correlation'].abs()
sorted_corr = corr_flat.sort_values(by='Abs_Correlation', ascending=False).head(10)
print("Top correlated feature pairs:")
print(sorted_corr)
sorted_corr.to_csv('top_correlated_pairs.csv', index=False)
print("Top correlated feature pairs saved to 'top_correlated_pairs.csv'.")

# Generate Q-Q plots
os.makedirs("qq_plots", exist_ok=True)
for column in non_categorical_variables:
    plt.figure(figsize=(6, 4))
    sm.qqplot(df[column].dropna(), line='45', fit=True)
    plt.title(f'Q-Q Plot for {column}')
    plt.savefig(f'qq_plots/qq_plot_{column}.png')
    plt.close()
    print(f"Q-Q plot saved for {column}")

# Generate scatter plots
os.makedirs("scatter_plots", exist_ok=True)
for i in range(len(non_categorical_variables)):
    for j in range(i + 1, len(non_categorical_variables)):
        var1 = non_categorical_variables[i]
        var2 = non_categorical_variables[j]
        plt.figure(figsize=(6, 4))
        plt.scatter(df[var1], df[var2], alpha=0.5)
        plt.title(f'Scatter Plot between {var1} and {var2}')
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.savefig(f'scatter_plots/scatter_{var1}_vs_{var2}.png')
        plt.close()
        print(f'Scatter plot saved for {var1} vs {var2}')


# Create a SQLAlchemy engine
engine = create_engine('mysql+pymysql://root:Yetayalmysql101!@localhost/data_eng_300_hw1')

# Save the cleaned DataFrame to a new table in the database
df.to_sql('heart_disease_cleaned', con=engine, index=False, if_exists='replace')

print("Data saved to 'heart_disease_cleaned' in the database.")
