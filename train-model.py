import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("seattle-weather.csv")

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.weekday

feature_cols =[
    "precipitation", "temp_max", "temp_min", "wind", "weather", "month", "weekday"
    ]
X_raw = df[feature_cols]

y = df[["precipitation", "temp_max", "temp_min", "wind"]].shift(-1)
X_raw = X_raw.iloc[:-1].copy()
y = y.iloc[:-1].copy()

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size = 0.2, shuffle = False)

numeric_features = ["precipitation", "temp_max", "temp_min", "wind"]
categorical_features = ["weather", "month", "weekday"]

preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), numeric_features), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)])

preprocessor.fit(X_train)
joblib.dump(preprocessor, "preprocessor_weather.joblib")
print("Preprocessor saved to file...")

X_train_pre = preprocessor.transform(X_train)
X_test_pre = preprocessor.transform(X_test)

#This gives us 4 outputs: precip, high, low, wind
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_pre.shape[1],)),
    Dense(64, activation="relu"),
    Dense(y_train.shape[1])
    ])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
    )

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
    )

history = model.fit(
    X_train_pre, y_train.values,
    validation_data=(X_test_pre, y_test.values),
    epochs=250,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
    )

model.save("model-weather.keras")
print("Model saved to file...")

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Training-Weather-Plot.png")
plt.show()