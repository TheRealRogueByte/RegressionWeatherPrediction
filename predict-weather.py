import sys
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

try:
    preprocessor = joblib.load("preprocessor_weather.joblib")
    model = load_model("model-weather.keras")
except Exception as e:
    print("Error:", e)
    sys.exit(1)
    
date_str = input("Enter today's date(YYYY-MM-DD): ").strip()
precip = float(input("Today's precipitation (e.g. mm): ").strip())
temp_max = float(input("Today's high temperature: ").strip())
temp_min = float(input("Today's low temperature: ").strip())
wind = float(input("Today's average wind speed: ").strip())
weather_cat = input("Today's weather category (e.g. rain, clear, hail): ").strip()

try:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
except ValueError:
    print("Invalid format")
    sys.exit(1)

month = dt.month
weekday = dt.weekday()

X_user = pd.DataFrame([{
    "precipitation": precip,
    "temp_max": temp_max,
    "temp_min": temp_min,
    "wind": wind,
    "weather": weather_cat,
    "month": month,
    "weekday": weekday
    }])

X_trans = preprocessor.transform(X_user)
y_pred = model.predict(X_trans)[0]

pred_precip, pred_high, pred_low, pred_wind = y_pred

print("\nForecast for Tomorrow")
print(f"  Precipitation: {pred_precip:.2f}")
print(f"  High Temp:     {pred_high:.2f}")
print(f"  Low Temp:      {pred_low:.2f}")
print(f"  Wind Speed:    {pred_wind:.2f}")