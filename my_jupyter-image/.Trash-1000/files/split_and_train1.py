import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('/Users/yetayaltizale/Documents/DataEng_300_HW2/heart_disease_cleaned.csv')

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
    'Decision Tree': DecisionTreeClassifier()
}

# Grid search parameters for tuning
param_grid = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
}

results = {}

# Train models and tune hyperparameters
for name, model in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    results[name] = {
        'best_params': grid_search.best_params_,
        'best_estimator': grid_search.best_estimator_,
        'cv_scores': cross_val_score(grid_search.best_estimator_, X_train_scaled, y_train, cv=5),
        'mean_cv_score': grid_search.best_score_
    }
    print(f"Best parameters for {name}: {results[name]['best_params']}")
    print(f"Mean CV Score for {name}: {results[name]['mean_cv_score']:.3f}")

# Evaluate on test data and find the best model
best_model_name = max(results, key=lambda x: results[x]['mean_cv_score'])
best_model = results[best_model_name]['best_estimator']

best_model.fit(X_train_scaled, y_train)
predictions = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print(f"\nBest Model: {best_model_name}")
print(f"Best Model Parameters: {results[best_model_name]['best_params']}")
print(f"Best Cross-Validation Score: {results[best_model_name]['mean_cv_score']:.3f}")
print(f"{best_model_name} - Test Set Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
