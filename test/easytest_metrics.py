import numpy as np
import pandas as pd
import warnings
import json
import matplotlib.pyplot as plt
from datasetsforecast.m3 import M3
from src.metrics import Pelt, STLFeatures, ACF_Features, CrossingPoints, LinearRegression
import statsmodels.api as sm
from ruptures.base import BaseEstimator
from ruptures.costs import cost_factory
from math import floor

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def test_easytest_metrics():
    """
    Analyzes M3 monthly data using custom classes from src/metrics.py.
    """
    # --- Load Data (Will crash if fails) ---
    data_dir = './data'
    print(f"\nLoading M3 Monthly data (will download to '{data_dir}' if needed)...")

    loaded_data = M3.load(directory=data_dir, group='Monthly')
    if isinstance(loaded_data, (list, tuple)): Y_df = loaded_data[0]
    elif isinstance(loaded_data, pd.DataFrame): Y_df = loaded_data
    else: raise TypeError("Loaded data not DataFrame")
    print("Data loaded successfully.")

    required_cols = ['unique_id', 'ds', 'y']
    if not all(col in Y_df.columns for col in required_cols):
        raise ValueError(f"Error: Missing required columns: {[c for c in required_cols if c not in Y_df.columns]}")
    # --- End Load Data ---

    all_dataset_ids = Y_df['unique_id'].unique()
    results_data = []

    # --- Instantiate custom metric classes (Will crash if fails) ---
    print("Instantiating custom metric classes (using Pelt library defaults)...")
    # Use default parameters for Pelt instance from custom class
    pelt_instance = Pelt() # Uses model="l2", min_size=2, jump=5 by default
    # Other custom classes
    stl_instance = STLFeatures(freq=12)
    acf_instance = ACF_Features(freq=12)
    crossing_points_instance = CrossingPoints(freq=12)
    linear_regression_instance = LinearRegression()
    # --- End Instantiation ---

    print(f"\nAnalyzing {len(all_dataset_ids)} monthly time series using CUSTOM metrics (NO error handling)...")

    processed_count = 0
    for i, dataset_id in enumerate(all_dataset_ids):
        df_series = Y_df[Y_df['unique_id'] == dataset_id].sort_values('ds')
        series_pd = df_series['y'].reset_index(drop=True)

        # --- Pre-computation Checks ---
        if series_pd.isnull().any() or not np.isfinite(series_pd).all(): continue
        if series_pd.nunique() <= 1: continue
        series_np = series_pd.to_numpy()
        min_len_overall = 2
        min_len_stl = 2 * 12 + 1
        min_len_pelt = pelt_instance.min_size
        if len(series_np) < max(min_len_overall, min_len_pelt): continue
        # --- End Pre-computation Checks ---

        processed_count += 1
        # Initialize features dict with CONSISTENT keys
        current_features = {
            "unique_id": dataset_id,
            "Pelt_Num_Breakpoints": np.nan,
            "STL_Trend_Strength": np.nan, # Consistent Key
            "ACF_FirstLag": np.nan,
            "CrossingPoints": np.nan,     # Consistent Key
            "LinearRegression_Slope": np.nan,
            "LinearRegression_R2": np.nan,
        }

        # --- Feature Extraction ---

        # 1. Pelt Breakpoints - Using fit/predict pattern, log(n) penalty
        pen_value = np.log(len(series_np))
        pelt_instance.fit(series_np)
        bkps = pelt_instance.predict(pen=pen_value)
        num_changepoints = len(bkps) - 1 if bkps else 0
        current_features["Pelt_Num_Breakpoints"] = max(0, num_changepoints)

        # 2. STL Features - Use custom get_features(x=...), store with consistent key
        if len(series_np) >= min_len_stl:
            stl_result = stl_instance.get_features(x=series_np)
            # *** Store under CONSISTENT KEY ***
            current_features["STL_Trend_Strength"] = stl_result.get('trend', np.nan)
        # else: STL_Trend_Strength remains NaN

        # 3. ACF Features - Use custom get_features(x=...)
        acf_result = acf_instance.get_features(x=series_np)
        current_features["ACF_FirstLag"] = acf_result.get('x_acf1', np.nan)

        # 4. Crossing Points - Use custom get_features(x=...), extract value, store with consistent key
        cp_result = crossing_points_instance.get_features(x=series_np.astype(float))
        # *** Store under CONSISTENT KEY ***
        current_features["CrossingPoints"] = cp_result.get('crossing_points', np.nan)

        # 5. Linear Regression - Use custom fit()/score()
        linear_regression_instance.fit(time_series=series_np)
        current_features["LinearRegression_Slope"] = linear_regression_instance.coef_
        current_features["LinearRegression_R2"] = linear_regression_instance.score()

        # --- Append results for this series ---
        results_data.append(current_features)

        # --- Progress Update ---
        if (processed_count) % 100 == 0 or (i + 1) == len(all_dataset_ids):
            print(f"Processed {i + 1}/{len(all_dataset_ids)} potential series... ({processed_count} completed)")

    # --- AFTER THE LOOP ---
    print(f"\nAnalysis loop complete. Features computed for {processed_count} series.")

    if not results_data:
        print("No data was processed successfully.")
        return {}

    # --- Convert results to DataFrame (Will crash if fails) ---
    print("Converting results to DataFrame...")
    results_df = pd.DataFrame(results_data)
    if 'unique_id' in results_df.columns:
         results_df.set_index('unique_id', inplace=True)
    else:
        print("Warning: 'unique_id' column missing after DataFrame creation.")

    # --- Display Results (Will crash if DataFrame is invalid) ---
    print("\n--- Custom Feature Calculation Summary (Pelt Defaults) ---")
    print(results_df.info())
    print("\n--- Custom Descriptive Statistics (Pelt Defaults) ---")
    with pd.option_context('display.float_format', '{:,.2f}'.format): print(results_df.describe())
    print("\n--- Sample of Custom Results DataFrame (Pelt Defaults) ---")
    print(results_df.head())
    print("\n--- Custom Extreme Values per Feature (Pelt Defaults) ---")
    extreme_results = {}
    # Direct calculation (Will crash on errors)
    for feature_name in results_df.select_dtypes(include=np.number).columns:
        valid_series = results_df[feature_name].dropna()
        if valid_series.empty:
            print(f"\nFeature: {feature_name}\n  No valid finite data found.")
            extreme_results[feature_name] = {'lowest': (None, None), 'highest': (None, None)}
            continue

        idx_min, val_min = valid_series.idxmin(), valid_series.min()
        idx_max, val_max = valid_series.idxmax(), valid_series.max()
        print(f"\nFeature: {feature_name}")
        print(f"  Lowest:  ID = {idx_min}, Value = {val_min:.4f}")
        print(f"  Highest: ID = {idx_max}, Value = {val_max:.4f}")
        extreme_results[feature_name] = {'lowest': (idx_min, val_min), 'highest': (idx_max, val_max)}

    return extreme_results

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if np.isnan(obj): return None
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    print(f"--- Running Custom Metrics Analysis Script: {__file__} ---")
    analysis_results = test_easytest_metrics()

    print("\nAnalysis function completed without crashing.")
    if analysis_results:
        print("Results dictionary was generated (showing extremes above).")
    else:
        print("Analysis generated no results.")