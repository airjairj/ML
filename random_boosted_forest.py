import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

# Load data
train_df = pd.read_csv("Contest1\\train.csv")
test_df = pd.read_csv("Contest1\\test.csv")

# Split features and labels
X_train = train_df.drop(columns=["row ID", "LANDCOVER"])
y_train = train_df["LANDCOVER"].astype(int)
X_test = test_df.drop(columns=["row ID"])

# Build a pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

# Tune the model
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [10, 20, None],
    "clf__min_samples_split": [2, 5],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best F1 macro: {grid.best_score_:.4f}")
print("Best parameters:", grid.best_params_)

# Predict and save output
y_pred = grid.predict(X_test)
submission = pd.DataFrame({
    "ID": test_df["row ID"],
    "LANDCOVER": y_pred
})
submission.to_csv("Contest1\\submission_rf_boosted.csv", index=False)
print("Random Forest submission saved as 'submission_rf_boosted.csv'")
# Note: The RandomForestClassifier is not a boosted model, but it can be used in a similar way to the SVM and Decision Tree classifiers.