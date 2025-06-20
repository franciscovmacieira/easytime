import numpy as np
from statsmodels.tsa.seasonal import STL
from ruptures.detection import Pelt 
import tsfeatures
from sklearn.linear_model import LinearRegression as SkLearnLinearRegression
import antropy as ant
import pycatch22
import tsfel

# Trend Analysis

def trend_strength(series, period=12, seasonal=7, robust=False): # Trend Strength
    stl_sm = STL(series, period=period, seasonal=seasonal, robust=robust)
    res_sm = stl_sm.fit()
    remainder = res_sm.resid; deseason = series - res_sm.seasonal
    vare = np.nanvar(remainder, ddof=1); vardeseason = np.nanvar(deseason, ddof=1)
    if vardeseason <= 1e-10: trend_strength = 0.0
    else: trend_strength = max(0., min(1., 1. - (vare / vardeseason if vardeseason > 1e-10 else 1e-10)))
    return trend_strength

def trend_changes(series, model="l2", custom_cost=None, min_size=2, jump=5, params=None): # Trend Changes
    pelt_instance = Pelt(model=model, custom_cost=custom_cost, min_size=min_size, jump=jump,params=params)
    pen_value = np.log(len(series)) if len(series) > 1 else 0
    algo = pelt_instance.fit(series)
    bkps = algo.predict(pen=pen_value)
    num_changepoints = len(bkps) - 1 if bkps else 0
    return num_changepoints

def linear_regression_slope(series): # Linear Regression Slope
    lr_model_instance = SkLearnLinearRegression()
    time_steps = np.arange(len(series)).reshape(-1, 1)
    lr_model_instance.fit(time_steps, series)
    slope = lr_model_instance.coef_[0] if lr_model_instance.coef_.size > 0 else np.nan
    return slope

def linear_regression_r2(series): # Linear Regression R2
    lr_model_instance = SkLearnLinearRegression()
    time_steps = np.arange(len(series)).reshape(-1, 1)
    lr_model_instance.fit(time_steps, series)
    r_squared = lr_model_instance.score(time_steps, series)
    return r_squared

# Noise/Complexity

def forecastability(series, sf, method="welch", nperseg=None, normalize=False): # Series Forecastabality
    spec_entropy = ant.spectral_entropy(series, sf=sf, method=method, nperseg=nperseg, normalize=normalize)
    return 1/spec_entropy

def fluctuation(series): # Series Fluctuation
    series_list = series.tolist()
    catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
    feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
    fluct_value = feature_dict.get('MD_hrv_classic_pnn40', np.nan)
    return fluct_value

# Seasonality Detection

def ac_relevance(series): # AutoCorrelation Relevance
    series_list = series.tolist()
    catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
    feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
    e_crossing = feature_dict.get('CO_f1ecac', np.nan)
    return e_crossing

def seasonal_strength(series, period=12, seasonal=7, robust=False): # Seasonal Strength
    stl_sm = STL(series, period=period, seasonal=seasonal, robust=robust)
    res_sm = stl_sm.fit()
    remainder = res_sm.resid
    trend_component = res_sm.trend
    detrended_series = series - trend_component
    var_remainder = np.nanvar(remainder, ddof=1)
    var_detrended = np.nanvar(detrended_series, ddof=1)
    if var_detrended <= 1e-10: seasonal_strength_val = 0.0 if var_remainder <= 1e-10 else 1.0
    else: seasonal_strength_val = max(0., min(1., 1. - (var_remainder / var_detrended)))
    return seasonal_strength_val

# Volatility/Outliers

def window_fluctuation(series): # Window Fluctuation
    series_list = series.tolist()
    catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
    feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
    fluct = feature_dict.get('SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', np.nan)
    return fluct

# Model Selection

def st_variation(series): # Short-Term Variation
    series_list = series.tolist()
    catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
    feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
    variation = feature_dict.get('CO_trev_1_num', np.nan)
    return variation

def diff_series(series): # Differenced Series
    result_dict = tsfeatures.acf_features(series)
    diff = result_dict['diff1_acf10']
    return diff

def complexity(series) -> float: # Series Complexity
    """
    Calculates the Complexity Estimate (CE) for the time series.
    This CE is a measure of the complexity of a single time series, as defined
    in "CID: An efficient complexity-invariant distance for time series".
    """
    series_np = np.asarray(series, dtype=np.float64)

    if series_np.ndim != 1:
        raise ValueError("Input series (series) must be 1-dimensional.")
    
    if len(series_np) < 2:
        raise ValueError(
            f"Series (series) must have at least 2 data points to calculate "
            f"its complexity estimate. Current length: {len(series_np)}"
        )

    # 1. Z-normalize the series
    mean_ts = np.mean(series_np)
    std_ts = np.std(series_np)
    if std_ts == 0:
        series_np = np.zeros_like(series_np, dtype=np.float64)
    series_np = (series_np - mean_ts) / std_ts

    # 2. Calculate Complexity Estimate (CE)
    if len(series_np) < 2:
        return 0.0 
    diff_ts = np.diff(series_np) 
    ce = np.sqrt(np.sum(diff_ts**2))

    return ce

# Clustering/Classification

def rec_concentration(series): # Records Concentration
    series_list = series.tolist()
    catch22_raw_results = pycatch22.catch22_all(series_list, catch24=False)
    feature_dict = dict(zip(catch22_raw_results['names'], catch22_raw_results['values']))
    concentration = feature_dict.get('DN_HistogramMode_10', np.nan)
    return concentration

def centroid(series, fs: int): # Series Centroid
    centroid_value = tsfel.feature_extraction.features.calc_centroid(series, fs)
    return float(centroid_value)

# Information

@staticmethod
def info(): #Information
    print("\nSmall description of the features." \
    "\nFor the full documentation see the library oficial website: https://franciscovmacieira.github.io/easyts/ or the library GitHub repository: https://github.com/franciscovmacieira/Deep-Time-Series-Analysis.git" \
    "\ntrend_strength: Computes the strength of a trend within the time-series." \
    "\nmedian_crosses: Counts the number of times the time-series crosses its median." \
    "\ntrend_changes: Detects the number of trend changes in the time-series." \
    "\nlinear_regression_slope: Computes the slope of a linear regression fitted to the time-series." \
    "\nlinear_regression_r2: Computes the R-squared value of a linear regression fitted to the time-series." \
    "\nforecastability: Measures the forecastability of the time-series using spectral entropy." \
    "\nentropy_pairs: Computes the entropy of the time-series." \
    "\nfluctuation: Measures the fluctuation of the time-series." \
    "\nac_relevance: Computes the autocorrelation relevance of the time-series." \
    "\nseasonal_strength: Computes the strength of seasonality within the time-series." \
    "\nwindow_fluctuation: Measures the proportion of fluctuations in short windows of the time-series." \
    "\nst_variation: Computes the short-term variation of the time-series." \
    "\nac: Computes the autocorrelation of the time-series." \
    "\ndiff_series: Computes the autocorrelation value of the differenced series." \
    "\ncomplexity: Computes the complexity of the time-series." \
    "\nrec_concentration: Computes the relative position of the most frequent values in relation to the mean." \
    "\ncentroid: Computes the centroid of the time-series.\n" \
    )

def all_metrics(series, sf):
    return trend_strength(series), \
            trend_changes(series), \
            linear_regression_slope(series), \
            linear_regression_r2(series), \
            forecastability(series, sf), \
            fluctuation(series), \
            ac_relevance(series), \
            seasonal_strength(series), \
            window_fluctuation(series), \
            st_variation(series), \
            diff_series(series), \
            complexity(series), \
            rec_concentration(series), \
            centroid(series, sf)
    





    

