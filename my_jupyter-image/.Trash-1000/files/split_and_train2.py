import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset
data = pd.read_csv('/Users/yetayaltizale/Documents/DataEng_300_HW2/heart_disease_cleaned.csv')

# Create interaction terms
data['age_trestbps'] = data['age'] * data['trestbps']
data['smoke_oldpeak'] = data['smoke'] * data['oldpeak']

# Preparing the data
X = data.drop(['target'], axis=1)
y = data['target']

# Splitting the data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

# Setup the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier())
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.01, 0.1]
}

# Grid search to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Evaluate on test data
predictions = grid_search.predict(X_test)

print("Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")
print(f"Classification Report:\n{classification_report(y_test, predictions)}")
