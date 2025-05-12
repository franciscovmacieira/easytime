import numpy as np
from statsmodels.tsa.seasonal import STL
from ruptures.detection import Pelt 
import tsfeatures
from sklearn.linear_model import LinearRegression as SkLearnLinearRegression
from statsmodels.tsa.seasonal import STL
import antropy as ant
import pycatch22
import ruptures as rpt
import tsfel

# Metrics Class

class Metrics ():
    def __init__(self, series):
        self.series = series
        self.series_len = len(series)
        self.series_list = self.series.tolist()

    # Trend Analysis

    def trend_strength(self, period=12, seasonal=7, robust=False): # Trend Strength
        stl_sm = STL(self.series, period=period, seasonal=seasonal, robust=robust)
        res_sm = stl_sm.fit()
        remainder = res_sm.resid; deseason = self.series - res_sm.seasonal
        vare = np.nanvar(remainder, ddof=1); vardeseason = np.nanvar(deseason, ddof=1)
        if vardeseason <= 1e-10: trend_strength = 0.0
        else: trend_strength = max(0., min(1., 1. - (vare / vardeseason if vardeseason > 1e-10 else 1e-10)))
        return trend_strength
    
    def median_crosses(self): # Median Crosses
        result_dict = tsfeatures.crossing_points(self.series)
        n_points = result_dict['crossing_points']
        return n_points
    
    def trend_changes(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None): # Trend Changes
        pelt_instance = Pelt(model=model, custom_cost=custom_cost, min_size=min_size, jump=jump,params=params)
        pen_value = np.log(self.series_len) if self.series_len > 1 else 0
        algo = pelt_instance.fit(self.series)
        bkps = algo.predict(pen=pen_value)
        num_changepoints = len(bkps) - 1 if bkps else 0
        return num_changepoints
    
    def linear_regression_slope(self): # Linear Regression Slope
        lr_model_instance = SkLearnLinearRegression()
        time_steps = np.arange(self.series_len).reshape(-1, 1)
        lr_model_instance.fit(time_steps, self.series)
        slope = lr_model_instance.coef_[0] if lr_model_instance.coef_.size > 0 else np.nan
        return slope
    
    def linear_regression_r2(self): # Linear Regression R2
        lr_model_instance = SkLearnLinearRegression()
        time_steps = np.arange(self.series_len).reshape(-1, 1)
        lr_model_instance.fit(time_steps, self.series)
        r_squared = lr_model_instance.score(time_steps, self.series)
        return r_squared
    
    # Noise/Complexity
    
    def forecastability(self, sf, method="welch", nperseg=None, normalize=False): # Series Forecastabality
        spec_entropy = ant.spectral_entropy(self.series, sf=sf, method=method, nperseg=nperseg, normalize=normalize)
        return spec_entropy
    
    def entropy_pairs(self): # Entropy Pairs
        series_list = self.series.tolist()
        catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
        feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values'])) 
        entropy_pairs = feature_dict.get('SB_MotifThree_quantile_hh', np.nan)
        return entropy_pairs
    
    def fluctuation(self): # Series Fluctuation
        series_list = self.series.tolist()
        catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
        feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
        fluct_value = feature_dict.get('MD_hrv_classic_pnn40', np.nan)
        return fluct_value
    
    # Seasonality Detection
    
    def ac_relevance(self): # AutoCorrelation Relevance
        series_list = self.series.tolist()
        catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
        feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
        e_crossing = feature_dict.get('CO_f1ecac', np.nan)
        return e_crossing
    
    def seasonal_strength(self, period=12, seasonal=7, robust=False): # Seasonal Strength
        stl_sm = STL(self.series, period=period, seasonal=seasonal, robust=robust)
        res_sm = stl_sm.fit()
        remainder = res_sm.resid
        trend_component = res_sm.trend
        detrended_series = self.series - trend_component
        var_remainder = np.nanvar(remainder, ddof=1)
        var_detrended = np.nanvar(detrended_series, ddof=1)
        if var_detrended <= 1e-10: seasonal_strength_val = 0.0 if var_remainder <= 1e-10 else 1.0
        else: seasonal_strength_val = max(0., min(1., 1. - (var_remainder / var_detrended)))
        return seasonal_strength_val
    
    # Volatility/Outliers
    
    def window_fluctuation(self): # Window Fluctuation
        series_list = self.series.tolist()
        catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
        feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
        fluct = feature_dict.get('SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', np.nan)
        return fluct
    
    # Model Selection
    
    def st_variation(self): # Short-Term Variation
        series_list = self.series.tolist()
        catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
        feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
        variation = feature_dict.get('CO_trev_1_num', np.nan)
        return variation
    
    def ac (self): # AutoCorrelation
        result_dict = tsfeatures.acf_features(self.series)
        ac = result_dict['x_acf1']
        return ac
    
    def diff_series(self): # Differenced Series
        result_dict = tsfeatures.acf_features(self.series)
        diff = result_dict['diff1_acf10']
        return diff
    

    # Helper functions to Series Complexity
    # ----------------------------------------------------------------------------------------------

    def z_normalize(self, series: np.ndarray) -> np.ndarray: # Z-Normalization
        mean = np.mean(series)
        std_dev = np.std(series)
        if std_dev < 1e-10:
            return np.zeros_like(series)
        else:
            return (series - mean) / std_dev

    def calculate_ce(self, series: np.ndarray) -> float: #Complexity Estimate
        if series.ndim != 1:
            raise ValueError("Input series must be 1-dimensional.")
        if len(series) < 2:
            return 0.0
        # Handle cases where the segment was constant before normalization
        if np.allclose(series, 0):
            return 0.0
        diffs = np.diff(series)
        ce_val = np.linalg.norm(diffs)
        return ce_val

    # ----------------------------------------------------------------------------------------------

    def complexity(self, window_size: int = 5, penalty_value: float = None, model: str = "rbf", min_size: int = 2) -> int:
        # --- Input Validation ---
        if not isinstance(self.series, np.ndarray):
            self.series = np.asarray(self.series, dtype=float)
        if self.series.ndim != 1:
            raise ValueError("Input self.series must be 1-dimensional.")
        n = len(self.series)
        if window_size < 2 or window_size > n:
            raise ValueError(f"window_size must be between 2 and len(self.series)={n}.")
        if n < window_size + min_size : # Need at least one full window and min_size samples after
            print(f"Warning: self.series length ({n}) is too short for window size ({window_size}) and min_size ({min_size}). Returning empty list.")
            return []

        # --- Calculate Rolling Complexity Estimate ---
        rolling_ce = []
        for i in range(n - window_size + 1):
            segment = self.series[i : i + window_size]
            # Normalize segment before calculating CE for robustness to local scale/offset
            segment_norm = self.z_normalize(segment)
            ce_val = self.calculate_ce(segment_norm)
            rolling_ce.append(ce_val)

        rolling_ce_np = np.array(rolling_ce)

        if len(rolling_ce_np) < min_size * 2: # Need enough points for change detection
            print(f"Warning: Rolling complexity self.series is too short ({len(rolling_ce_np)}) for min_size ({min_size}). Returning empty list.")
            return []

        # --- Detect Change Points in Rolling CE ---
        try:
            n_ce = len(rolling_ce_np)
            if penalty_value is None:
                sigma = np.std(rolling_ce_np)
                if sigma < 1e-8: # Handle constant CE self.series
                    print("Warning: Rolling complexity is constant. No change points will be detected.")
                    return []
                penalty_value = 3 * np.log(n_ce) # Default penalty value

            algo = rpt.Pelt(model=model, min_size=min_size).fit(rolling_ce_np)
            change_points_ce_indices = algo.predict(pen=penalty_value)

            # Remove the last point which is always the length of the signal
            if len(change_points_ce_indices) > 0 and change_points_ce_indices[-1] == n_ce:
                change_points_ce_indices = change_points_ce_indices[:-1]

        except ImportError:
            print("Error: The 'ruptures' library is required for change point detection.")
            print("Please install it: pip install ruptures")
            raise # Re-raise the import error
        except Exception as e:
            print(f"Error during change point detection: {e}")
            return [] # Return empty list on error

        indices = [idx + window_size - 1 for idx in change_points_ce_indices]

        return len(indices)
    
    # Clustering/Classification

    def rec_concentration(self): # Records Concentration
        series_list = self.series.tolist()
        catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
        feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
        concentration = feature_dict.get('DN_HistogramMode_10', np.nan)
        return concentration
    
    def centroid(self, fs: int): # Series Centroid
        centroid_value = tsfel.feature_extraction.features.calc_centroid(self.series, fs)
        return float(centroid_value)

    





    

