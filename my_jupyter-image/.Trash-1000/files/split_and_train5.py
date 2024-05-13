import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load your dataset
data = pd.read_csv('/Users/yetayaltizale/Documents/DataEng_300_HW2/heart_disease_cleaned.csv')

# Apply transformations to skewed variables
data['log_smoke'] = np.log1p(data['smoke'])  # Assuming 'smoke' was identified as skewed
data['sqrt_oldpeak'] = np.sqrt(data['oldpeak'])  # Assuming 'oldpeak' was skewed

# Create interaction terms for highly correlated variables
data['age_trestbps'] = data['age'] * data['trestbps']  # Example of creating an interaction term

# Preparing the data
X = data.drop(['target'], axis=1)
y = data['target']

# Splitting the data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Grid search parameters for tuning
param_grid = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
}

# Train models and tune hyperparameters
best_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best CV score for {name}: {grid_search.best_score_:.3f}")

# Evaluate on test data and save the best model
for name, model in best_models.items():
    predictions = model.predict(X_test_scaled)
    print(f"\n{name} - Test Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")
    print(f"Classification Report:\n{classification_report(y_test, predictions)}")
    # Save the model
    joblib.dump(model, f'{name}_model.pkl')

# Identifying the best overall model
best_overall_model = max(best_models, key=lambda k: best_models[k].score(X_test_scaled, y_test))
print(f"Best Model: {best_overall_model}")
