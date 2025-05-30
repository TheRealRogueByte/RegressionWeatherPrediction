# Weather Forecasting (Regression) 🚀

This project uses a neural-network regression model to predict tomorrow’s weather variables (precipitation, high/low temperature, and wind speed) based on historical daily observations. It leverages time-series features, meteorological measurements, and categorical weather codes to provide accurate continuous forecasts.

## 🌤️ What It Does

- Predicts **Tomorrow’s Precipitation**, **High Temp**, **Low Temp**, and **Wind Speed**  
- Trains on historical data from `seattle-weather.csv`  
- Saves a serialized preprocessor and Keras model for easy reuse  
- Generates a **Training vs Validation Loss** plot to evaluate performance  

## 🧠 Model Details

- **Framework**: Keras (TensorFlow backend)  
- **Architecture**:  
  - Input layer with size = number of preprocessed features  
  - Dense(128, ReLU) → Dense(64, ReLU) → Dense(4, linear)  
- **Loss**: Mean Squared Error (MSE)  
- **Metrics**: Mean Absolute Error (MAE)  
- **Optimizer**: Adam  
- **EarlyStopping**: monitor `val_loss`, patience = 20  

## 🛠️ Preprocessing Pipeline

- **Temporal Features**:  
  - Parse `date` → extract `month` and `weekday`  
- **Numeric Features**:  
  - `precipitation`, `temp_max`, `temp_min`, `wind` → standardized with `StandardScaler`  
- **Categorical**:  
  - `weather`, `month`, `weekday` → one-hot encoded  
- All transforms are combined in a `ColumnTransformer` and saved as `preprocessor_weather.joblib`.

## 📁 Files

- **`train_weather_model.py`**  
  - Loads `seattle-weather.csv`  
  - Builds & fits the preprocessing pipeline → saves `preprocessor_weather.joblib`  
  - Splits data (80/20, no shuffle)  
  - Defines & trains Keras model → saves `model_weather.keras`  
  - Outputs `training_weather_plot.png`
  
- **`predict_weather.py`**  
  - Loads `preprocessor_weather.joblib` & `model_weather.keras`  
  - Prompts for today’s weather inputs  
  - Transforms inputs and runs `model.predict`  
  - Prints tomorrow’s forecast

- **Data**  
  - `seattle-weather.csv` — historical daily observations (date, precipitation, temp_max, temp_min, wind, weather)

## ▶️ How to Run

1. **Install dependencies**  
   ```bash
   pip install pandas numpy scikit-learn matplotlib tensorflow joblib

2. **Train the model**
   ```bash
   python train_weather_model.py
4. **Run prediction**
   ```bash
   python predict_weather.py
5. **Follow CLI prompts for today’s date, precipitation, high/low temp, wind speed, and weather category.**

## Example Output
````
Forecast for Tomorrow
  Precipitation:  2.35 mm
  High Temp:     18.42 °C
  Low Temp:      11.78 °C
  Wind Speed:     3.21 m/s
````
