import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import early_stopping, log_evaluation

# ========== 1. Caricamento dei dati ==========
df_train = pd.read_csv("Contest2\\train.csv")
df_test = pd.read_csv("Contest2\\test.csv")

# ========== 2. Feature Engineering ==========
def preprocess(df, is_train=True, train_columns=None):
    df = df.copy()
    df['UTC'] = pd.to_datetime(df['UTC'])
    df['hour'] = df['UTC'].dt.hour
    df['month'] = df['UTC'].dt.month
    df['day'] = df['UTC'].dt.day

    # One-hot encoding
    df = pd.get_dummies(df, columns=["PLANT"])

    drop_cols = ['rowID', 'UTC']
    if 'Irr' in df.columns:
        drop_cols.append('Irr')
    df = df.drop(columns=drop_cols)

    # Per il test, riempi le colonne mancanti e ordina
    if not is_train and train_columns is not None:
        for col in train_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[train_columns]

    return df

# Prima preprocessi il training
X = preprocess(df_train, is_train=True)
y = df_train['Irr']

# Salvi le colonne
train_columns = X.columns.tolist()

# Poi preprocessi il test usando le stesse colonne
X_test = preprocess(df_test, is_train=False, train_columns=train_columns)

# ========== 3. Split train/validation ==========
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 4. Addestramento LightGBM ==========
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(100)
    ]
)

# ========== 5. Valutazione ==========
y_pred_val = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {rmse:.4f}")

# ========== 6. Previsione sul test set ==========
y_pred_test = model.predict(X_test)

# ========== 7. Scrittura del file di sottomissione ==========
submission = pd.DataFrame({
    'rowID': df_test['rowID'],
    'Irr': y_pred_test
})

submission.to_csv("Contest2\\LightGBMsubmission.csv", index=False)
print("\LightGBMsubmission.csv salvato correttamente!")
