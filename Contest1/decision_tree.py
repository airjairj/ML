import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv("Contest1\\train.csv")
test_df = pd.read_csv("Contest1\\test.csv")

# Features and target
X = train_df.drop(columns=["row ID", "LANDCOVER"])
y = train_df["LANDCOVER"].astype(int)
X_test = test_df.drop(columns=["row ID"])

# Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Decision Tree with class weights
clf = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=20)
clf.fit(X_train, y_train)

# Evaluate
y_val_pred = clf.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"Validation F1 Macro (Decision Tree, weighted): {val_f1:.4f}")

# Predict test set
y_pred = clf.predict(X_test_scaled)

# Save submission
submission = pd.DataFrame({
    "ID": test_df["row ID"],
    "LANDCOVER": y_pred
})
submission.to_csv("Contest1\\submission_tree_weighted.csv", index=False)
print("Decision Tree submission saved as 'submission_tree_weighted.csv'")
