import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Ensure the directory for output exists
output_dir = 'model_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
data = pd.read_csv('/Users/yetayaltizale/Documents/DataEng_300_HW2/heart_disease_cleaned.csv')

# Applying transformations
data['sex'] = np.where(data['sex'] == 0, 0, 1)  # Ensuring binary encoding is correct
data['smoke'] = np.log1p(data['smoke'])
data['fbs'] = np.log1p(data['fbs'])
data['prop'] = np.log1p(data['prop'])
data['nitr'] = np.log1p(data['nitr'])
data['pro'] = np.log1p(data['pro'])
data['diuretic'] = np.log1p(data['diuretic'])
data['oldpeak'] = np.sqrt(data['oldpeak'])
data['cdc_smoke_rate'] = np.sqrt(data['cdc_smoke_rate'])

# High correlation interaction terms
data['age_trestbps'] = data['age'] * data['trestbps']
data['smoke_oldpeak'] = data['smoke'] * data['oldpeak']

# Split data into features and target
X = data.drop(['target'], axis=1)
y = data['target']

# Splitting the data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

# Handling imbalanced data for the 'sex' variable
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# Predict on test data
predictions = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

# Save results
with open(f'{output_dir}/logistic_regression_results.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write(f"Classification Report:\n{class_report}\n")

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
