import numpy as np
from statsmodels.tsa.seasonal import STL
from ruptures.detection import Pelt 
import tsfeatures
from sklearn.linear_model import LinearRegression as SkLearnLinearRegression
from statsmodels.tsa.seasonal import STL
import antropy as ant
import pycatch22

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


    

