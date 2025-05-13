import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ========== 1. Caricamento ==========
df_train = pd.read_csv("Contest2\\train.csv")
df_test = pd.read_csv("Contest2\\test.csv")

# ========== 2. Preprocessing ==========
def preprocess(df, is_train=True, train_columns=None):
    df = df.copy()
    df['UTC'] = pd.to_datetime(df['UTC'])
    df['hour'] = df['UTC'].dt.hour
    df['month'] = df['UTC'].dt.month
    df['day'] = df['UTC'].dt.day

    # One-hot su PLANT
    df = pd.get_dummies(df, columns=['PLANT'])

    drop_cols = ['rowID', 'UTC']
    if 'Irr' in df.columns:
        drop_cols.append('Irr')
    df = df.drop(columns=drop_cols)

    if not is_train and train_columns is not None:
        for col in train_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[train_columns]

    return df

X = preprocess(df_train, is_train=True)
y = df_train['Irr']
train_columns = X.columns.tolist()
X_test = preprocess(df_test, is_train=False, train_columns=train_columns)

# ========== 3. Scaling ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ========== 4. Train/Validation split ==========
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== 5. SVR training ==========
model = SVR(kernel='rbf', C=10.0, epsilon=0.1)
model.fit(X_train, y_train)

# ========== 6. Validazione ==========
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse:.4f}")

# ========== 7. Predizione su test set ==========
y_test_pred = model.predict(X_test_scaled)

# ========== 8. Salvataggio submission ==========
submission = pd.DataFrame({
    'rowID': df_test['rowID'],
    'Irr': y_test_pred
})
submission.to_csv("Contest2\\SVR_submission.csv", index=False)
print("submission.csv salvato!")
