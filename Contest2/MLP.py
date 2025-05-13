import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor, early_stopping  # add early_stopping

# 1) load training data
train = pd.read_csv(r"Contest2\train.csv", parse_dates=["UTC"])

# 2) engineer cyclical features from datetime
# cyclical hour
train["hour"] = train.UTC.dt.hour
train["hour_sin"] = np.sin(2 * np.pi * train.hour / 24)
train["hour_cos"] = np.cos(2 * np.pi * train.hour / 24)
# day of year
train["dayoy"] = train.UTC.dt.dayofyear
train["doy_sin"] = np.sin(2 * np.pi * train.dayoy / 365)
train["doy_cos"] = np.cos(2 * np.pi * train.dayoy / 365)

# 3) drop identifiers & non-numeric fields
#    we drop:
#      - rowID    (just an index)
#      - UTC      (datetime string, could engineer features but drop for now)
#      - PLANT    (always plant_1 here; if multiple plants you’d one-hot)
#      - hour, dayoy (already encoded as cyclical features)
X = train.drop(columns=["rowID", "UTC", "PLANT", "Irr", "hour", "dayoy"])
y = train["Irr"].values  # continuous target

# 4) split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) build a Pipeline: scale → MLPRegressor
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    ))
])

# 6) fit
pipe.fit(X_train, y_train)

# 7) evaluate
y_pred = pipe.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# 7b) train LightGBM via sklearn API
X = train.drop(columns=["rowID", "UTC", "PLANT", "Irr"])
y = train.Irr.values
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

gbm = LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=1000,
    random_state=42
)
gbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=50)],  # updated with early_stopping
)

y_pred_lgb = gbm.predict(X_val)
print("LightGBM RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_lgb)))

# 8) load test set and make predictions
test = pd.read_csv(r"Contest2\test.csv", parse_dates=["UTC"])

# engineer cyclical features for test set
test["hour"] = test.UTC.dt.hour
test["hour_sin"] = np.sin(2 * np.pi * test.hour / 24)
test["hour_cos"] = np.cos(2 * np.pi * test.hour / 24)
test["dayoy"] = test.UTC.dt.dayofyear
test["doy_sin"] = np.sin(2 * np.pi * test.dayoy / 365)
test["doy_cos"] = np.cos(2 * np.pi * test.dayoy / 365)

X_test = test.drop(columns=["rowID", "UTC", "PLANT", "hour", "dayoy"])  # same drops as above
preds = pipe.predict(X_test)

# 9) write submission
submission = pd.DataFrame({
    "rowID": test["rowID"],
    "Irr":   preds
})
submission.to_csv("Contest2\\MLP_submission2.csv", index=False)