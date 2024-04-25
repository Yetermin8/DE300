import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
file_path = "/Users/yetayaltizale/Documents/heart_disease.xlsx"
df = pd.read_excel(file_path)

# Get the count of null values for each feature
null_counts = df.isnull().sum()

# Sort the features based on the count of null values in descending order
null_counts_sorted = null_counts[null_counts > 0].sort_values(ascending=False)

# Print the features with the most null values
print("Features with the most null values:")
print(null_counts_sorted)

# Save the plot as PNG image
null_counts_sorted.plot(kind='bar', figsize=(10, 6))
plt.title('Features with the most null values')
plt.xlabel('Features')
plt.ylabel('Count of Null Values')
plt.tight_layout()
plt.savefig('Features_Null_Value_Counts.png', dpi=300)
plt.show()
