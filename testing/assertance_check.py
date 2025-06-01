import numpy as np
import pandas as pd
import os
import sys

# --- Path and Module Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added to sys.path: {PROJECT_ROOT}")

# --- Required Imports ---
from datasetsforecast.m3 import M3
from src.metrics import Metrics

# --- Data Loading Helper ---

def _load_test_series():
    """
    Loads and prepares all valid series from the M3 Monthly dataset
    to be used for testing.
    """
    print("--- Loading M3 Monthly data ---")
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'm3_download')
    os.makedirs(data_dir, exist_ok=True)

    try:
        loaded_data = M3.load(directory=data_dir, group='Monthly')
    except Exception as e:
        print(f"Failed to load M3 data: {e}")
        return []

    Y_df = None
    if isinstance(loaded_data, (list, tuple)) and len(loaded_data) > 0:
        Y_df = loaded_data[0]
    elif isinstance(loaded_data, pd.DataFrame):
        Y_df = loaded_data

    if Y_df is None:
        raise ValueError("Failed to extract DataFrame from loaded M3 data.")

    if 'ds' not in Y_df.columns:
        Y_df = Y_df.rename(columns={'Series': 'unique_id'})
        Y_df = Y_df.melt(id_vars='unique_id', var_name='ds', value_name='y').dropna()
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    print(f"Loaded data for {Y_df['unique_id'].nunique()} series.")

    all_dataset_ids = Y_df['unique_id'].unique()
    series_to_test = []

    # Process all series from the dataset
    for dataset_id in all_dataset_ids:
        series_pd = Y_df[Y_df['unique_id'] == dataset_id]['y'].reset_index(drop=True)

        # Basic data quality checks
        if series_pd.isnull().any() or not np.isfinite(series_pd).all():
            continue
        if series_pd.nunique() <= 1 or len(series_pd) < 25:
            continue

        series_to_test.append(series_pd.to_numpy())

    print(f"Prepared {len(series_to_test)} valid series for testing.\n")
    return series_to_test

# --- Individual Metric Test Functions ---

def test_trend_strength():
    """Tests the trend_strength metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.trend_strength()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert 0.0 <= result <= 1.0, "Trend strength must be between 0 and 1"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_trend_strength passed.")

def test_median_crosses():
    """Tests the median_crosses metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.median_crosses()
        assert isinstance(result, (int, np.integer)), "Result must be an integer"
        assert result >= 0, "Median crosses cannot be negative"
    print("test_median_crosses passed.")

def test_trend_changes():
    """Tests the trend_changes metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.trend_changes()
        assert isinstance(result, (int, np.integer)), "Result must be an integer"
        assert result >= 0, "Number of trend changes cannot be negative"
    print("test_trend_changes passed.")

def test_linear_regression_slope():
    """Tests the linear_regression_slope metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.linear_regression_slope()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_linear_regression_slope passed.")

def test_linear_regression_r2():
    """Tests the linear_regression_r2 metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.linear_regression_r2()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert result <= 1.0, "R-squared cannot be greater than 1"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_linear_regression_r2 passed.")

def test_forecastability():
    """Tests the forecastability metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.forecastability(sf=1)
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert result >= 0, "Forecastability cannot be negative"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_forecastability passed.")

def test_entropy_pairs():
    """Tests the entropy_pairs metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.entropy_pairs()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_entropy_pairs passed.")

def test_fluctuation():
    """Tests the fluctuation metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.fluctuation()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_fluctuation passed.")

def test_ac_relevance():
    """Tests the ac_relevance metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.ac_relevance()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_ac_relevance passed.")

def test_seasonal_strength():
    """Tests the seasonal_strength metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.seasonal_strength()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert 0.0 <= result <= 1.0, "Seasonal strength must be between 0 and 1"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_seasonal_strength passed.")

def test_window_fluctuation():
    """Tests the window_fluctuation metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.window_fluctuation()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_window_fluctuation passed.")

def test_st_variation():
    """Tests the st_variation metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.st_variation()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_st_variation passed.")

def test_ac():
    """Tests the ac (autocorrelation) metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.ac()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert -1.0 <= result <= 1.0, "Autocorrelation must be between -1 and 1"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_ac passed.")

def test_diff_series():
    """Tests the diff_series metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.diff_series()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert result >= 0, "Sum of squared ACFs cannot be negative"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_diff_series passed.")

def test_complexity():
    """Tests the complexity metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.complexity()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert result >= 0.0, "Complexity cannot be negative"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_complexity passed.")

def test_rec_concentration():
    """Tests the rec_concentration metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.rec_concentration()
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_rec_concentration passed.")

def test_centroid():
    """Tests the centroid metric."""
    series_list = _load_test_series()

    for series in series_list:
        metrics = Metrics(series)
        result = metrics.centroid(fs=12)
        assert isinstance(result, (float, np.floating)), "Result must be a float"
        assert result >= 0.0, "Centroid cannot be negative"
        assert np.isfinite(result), "Result must be a finite number"
    print("test_centroid passed.")


# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Running All Metric Tests on the Full M3 Monthly Dataset ---")
    
    # Trend Analysis
    test_trend_strength()
    test_median_crosses()
    test_trend_changes()
    test_linear_regression_slope()
    test_linear_regression_r2()

    # Noise/Complexity
    test_forecastability()
    test_entropy_pairs()
    test_fluctuation()
    test_complexity()

    # Seasonality Detection
    test_ac_relevance()
    test_seasonal_strength()

    # Volatility/Outliers
    test_window_fluctuation()

    # Model Selection
    test_st_variation()
    test_ac()
    test_diff_series()
    
    # Clustering/Classification
    test_rec_concentration()
    test_centroid()
    
    print("\n--- All tests completed successfully! ---")