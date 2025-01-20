import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Load data
combined_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
)

X = combined_features.drop(columns=["label", "Sample Start", "Sample End"])  # Features

X = X.iloc[:, 1:]  # Remove the index column

column_names = list(X.columns.values)
column_names.sort()
X = X[column_names]

y = combined_features["label"]  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define hyperparameters for Random Forest
rf_params = {
    "n_estimators": [50, 100, 150],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"],
}


# Perform Grid Search for Random Forest
rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0),
    n_jobs=20,
    param_grid=rf_params,
    cv=5,
)
rf_grid_search.fit(X_train, y_train)

print(rf_grid_search.best_params_)

y_pred_rf_gs = rf_grid_search.predict(X_test)
print("\nRandom Forest Grid Search Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_gs))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf_gs))
