import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
import jsonpickle
import sys

# --- Configuration ---
ROUND_NUM = 4
# Use all available historical days for training
DAY_NUM = [1,2,3]
PRODUCT = 'MAGNIFICENT_MACARONS'
# Features to use for prediction (must match columns available after merging)
# Ensure these are the columns you want to use as predictors
FEATURE_COLS = ['mid_price', 'transportFees', 'exportTariff', 'importTariff', 'sugarPrice', 'sunlightIndex'] # <-- ADD 'mid_price' HERE
LAG_TARGET = -1 # Predict next timestamp's mid_price
OUTPUT_PARAM_FILE = 'round4/macaron_model_params.json' # File to save parameters

# --- File Paths ---
base_path = "sample_data/round-4-island-data-bottle/".format(ROUND_NUM)
prices_files = [f"{base_path}prices_round_{ROUND_NUM}_day_{day}.csv" for day in DAY_NUM]
observation_files = [f"{base_path}observations_round_{ROUND_NUM}_day_{day}.csv" for day in DAY_NUM]

# --- Load Data Function (Combined Price and Observations) ---
def load_and_merge_data(prices_files, observation_files, product_name):
    """Loads price and observation data, merges them for the specified product."""
    # Load Price Data
    all_prices_df = []
    for prices_file in prices_files:
        try:
            df = pd.read_csv(prices_file, sep=';')
            df['day'] = int(prices_file.split('_')[-1].split('.')[0])
            all_prices_df.append(df)
        except FileNotFoundError:
            print(f"Warning: Price file not found at {prices_file}. Skipping.")
        except Exception as e:
            print(f"Error loading price data from {prices_file}: {e}")
            return None
    if not all_prices_df: return None
    prices_df = pd.concat(all_prices_df, ignore_index=True)
    prices_df = prices_df[prices_df['product'] == product_name][['day', 'timestamp', 'mid_price']].copy()
    if prices_df.empty:
        print(f"Error: No data found for product '{product_name}' in price files.")
        return None
    prices_df['mid_price'] = pd.to_numeric(prices_df['mid_price'], errors='coerce')
    prices_df.set_index(['day', 'timestamp'], inplace=True)

    # Load Observation Data
    all_observations_df = []
    for observation_file in observation_files:
        try:
            df = pd.read_csv(observation_file, sep=',') # Comma separator for observations
            df['day'] = int(observation_file.split('_')[-1].split('.')[0])
            all_observations_df.append(df)
        except FileNotFoundError:
            print(f"Warning: Observation file not found at {observation_file}. Skipping.")
        except Exception as e:
            print(f"Error loading observation data from {observation_file}: {e}")
            return None
    if not all_observations_df: return None
    observations_df = pd.concat(all_observations_df, ignore_index=True)
    # Keep only relevant observation columns + index cols
    obs_cols_to_keep = ['day', 'timestamp'] + [col for col in FEATURE_COLS if col in observations_df.columns and col != 'mid_price']
    observations_df = observations_df[obs_cols_to_keep].copy()
    for col in observations_df.columns:
        if col not in ['day', 'timestamp']:
            observations_df[col] = pd.to_numeric(observations_df[col], errors='coerce')
    observations_df.set_index(['day', 'timestamp'], inplace=True)

    # Merge
    merged_df = pd.merge(prices_df, observations_df, left_index=True, right_index=True, how='inner')
    merged_df.sort_index(inplace=True)
    return merged_df

# --- Main Training Logic ---
if __name__ == "__main__":
    print(f"Loading data for {PRODUCT}...")
    data = load_and_merge_data(prices_files, observation_files, PRODUCT)

    if data is None or data.empty:
        print("Failed to load or merge data. Exiting.")
        sys.exit(1)

    print("Data loaded successfully.")

    # --- Feature Engineering ---
    print("Preparing features and target...")
    # Target: Predict the mid_price of the next timestamp
    data['target_mid_price'] = data['mid_price'].shift(LAG_TARGET)

    # Drop rows where target is NaN (last row(s)) or any feature is NaN
    # Ensure 'mid_price' itself doesn't have NaNs in the rows used for features
    data = data.dropna(subset=FEATURE_COLS + ['target_mid_price']) # Check NaNs in features AND target

    # Verify all feature columns exist
    actual_features = [col for col in FEATURE_COLS if col in data.columns]
    if len(actual_features) != len(FEATURE_COLS):
        print(f"Warning: Not all specified features found. Using: {actual_features}")
        missing = set(FEATURE_COLS) - set(actual_features)
        print(f"Missing features: {missing}")
        if not actual_features:
            print("Error: No features available for training. Exiting.")
            sys.exit(1)
    else:
        print(f"Using features: {actual_features}")

    X = data[actual_features]
    y = data['target_mid_price']

    if X.empty or y.empty:
        print("Error: No data available for training after processing. Exiting.")
        sys.exit(1)

    # --- Model Training ---
    print("Training Bayesian Ridge model...")
    model = BayesianRidge(compute_score=True) # compute_score can be useful for analysis
    try:
        model.fit(X, y)
    except Exception as e:
        print(f"Error during model fitting: {e}")
        sys.exit(1)

    print("Model training complete.")

    # --- Evaluate (Optional) ---
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"\nTraining Mean Squared Error: {mse}")
    print(f"R^2 Score on Training Data: {model.score(X, y)}") # R^2 score

    # --- Extract and Save Parameters ---
    intercept = model.intercept_
    coefficients = model.coef_
    feature_names = actual_features # Use the list of features actually used

    print("\n--- Extracted Model Parameters ---")
    print(f"Intercept: {intercept}")
    print("Coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"  {feature}: {coef}")

    # Package parameters for saving
    model_params = {
        'intercept': intercept,
        'coefficients': coefficients.tolist(), # Convert numpy array to list for jsonpickle
        'feature_names': feature_names
    }

    print(f"\nSaving parameters to {OUTPUT_PARAM_FILE}...")
    try:
        frozen = jsonpickle.encode(model_params)
        with open(OUTPUT_PARAM_FILE, 'w') as f:
            f.write(frozen)
        print("Parameters saved successfully.")
    except Exception as e:
        print(f"Error saving parameters: {e}")


import plotly.express as px

# Reset index to make 'day' and 'timestamp' regular columns
data_reset = data.reset_index()

# Build a DataFrame for plotting using the reset index DataFrame
plot_df = pd.DataFrame({
    'timestamp': data_reset['timestamp'], # Use the 'timestamp' column from reset data
    'actual': y.values,                   # Use .values to align with the reset index
    'predicted': y_pred                   # y_pred is already a numpy array
})
# No need to sort by timestamp here if data_reset was already sorted,
# but sorting doesn't hurt if you want to be sure.
plot_df = plot_df.sort_values('timestamp')

# 1) Time‑series plot
fig_ts = px.line(
    plot_df,
    x='timestamp',
    y=['actual','predicted'],
    title='Actual vs Predicted Mid Price Over Time',
    labels={'value':'Mid Price','timestamp':'Timestamp','variable':'Series'}
)
fig_ts.show()

# 2) Scatter plot with OLS fit
fig_scatter = px.scatter(
    x=plot_df['actual'],   # Use the column from plot_df
    y=plot_df['predicted'], # Use the column from plot_df
    trendline='ols',
    labels={'x':'Actual Mid Price','y':'Predicted Mid Price'},
    title='Actual vs Predicted Mid Price Scatter'
)
fig_scatter.show()
