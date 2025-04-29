import numpy as np
import pandas as pd # Used for qcut in EntropyPairs helper
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL # Foundational: Kept for STLFeatures
from statsmodels.tsa.stattools import acf # Foundational: Kept for ACF_Features
from statsmodels.nonparametric._smoothers_lowess import lowess # Used in STLFeatures
from statsmodels.regression.linear_model import OLS, RegressionResults # Foundational: Used by LinearRegression
from statsmodels.tools.tools import add_constant
from typing import Dict, List, Optional
from math import floor, log2 # Used by Pelt, Entropy, SpectralEntropy
import matplotlib.pyplot as plt # For LinearRegression plot_fit method
from collections import Counter # For EntropyPairs
import sys
from scipy.signal import periodogram, welch
from scipy.stats import zscore # Import zscore function
from ruptures.base import BaseEstimator, BaseCost
from ruptures.costs import cost_factory
import ctypes
import numpy.ctypeslib as npct
import os

# --- Load Compiled C Library ---
C_LIB_LOADED = False
c_lib = None
try:
    # Determine library path and extension based on OS
    lib_name = None
    if sys.platform.startswith('win'):
        lib_name = 'c_metrics.dll'
    elif sys.platform.startswith('darwin'): # macOS
        lib_name = 'c_metrics.dylib'
    else: # Linux/other Unix-like
        lib_name = 'c_metrics.so'

    lib_path = os.path.join(os.path.dirname(__file__), lib_name)

    if not os.path.exists(lib_path):
        # Fallback: check current working directory
        lib_path_cwd = os.path.join(os.getcwd(), lib_name)
        if os.path.exists(lib_path_cwd):
            lib_path = lib_path_cwd
        else:
             raise OSError(f"Shared library '{lib_name}' not found in script directory or CWD.")

    c_lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded C library: {lib_path}")

    # --- Define C function signatures ---
    # 1. MD_hrv_classic_pnn40 (for HighFluctuation)
    c_pnn40_func = c_lib.MD_hrv_classic_pnn40
    c_pnn40_func.argtypes = [npct.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), ctypes.c_int]
    c_pnn40_func.restype = ctypes.c_double

    # 2. SB_MotifThree_quantile_hh (for EntropyPairs)
    c_motif3_func = c_lib.SB_MotifThree_quantile_hh
    c_motif3_func.argtypes = [npct.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), ctypes.c_int]
    c_motif3_func.restype = ctypes.c_double

    C_LIB_LOADED = True

except (OSError, AttributeError, Exception) as e:
    print(f"--- CRITICAL WARNING ---")
    print(f"Failed to load or configure C library functions from '{lib_name}': {e}")
    print(f"Features 'HighFluctuation' and 'EntropyPairs' will return NaN.")
    print(f"Ensure '{lib_name}' is compiled correctly and placed next to metrics.py or in CWD.")
    print(f"--- END WARNING ---")
    c_pnn40_func = None
    c_motif3_func = None

# --- Helper Functions ---

def _xlogx_for_entropy(p: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Computes p * log_b(p) element-wise, returning 0 for p=0 elements.
    Used for Shannon entropy calculation: H = -sum(_xlogx_for_entropy(p)).
    """
    # (Implementation from previous version)
    p = np.asarray(p); out = np.zeros_like(p, dtype=float); mask = p > 1e-12
    if mask.sum() == 0: return out
    if base == 2: log_func = np.log2
    elif base == np.e: log_func = np.log
    else: log_func = lambda x: np.log(x) / np.log(base)
    logs = np.full_like(p, -np.inf); logs[mask] = log_func(p[mask]); out[mask] = p[mask] * logs[mask]
    out[~mask] = 0.0; out[np.isnan(p)] = np.nan
    return out

def poly(t: np.ndarray, degree: int) -> np.ndarray:
    """Computes polynomial features for a time vector t."""
    # (Implementation from previous version)
    if degree < 0: raise ValueError("Degree cannot be negative.");
    if t.ndim != 1: raise ValueError("Input t must be 1D."); n = len(t); X = np.empty((n, degree + 1), dtype=t.dtype)
    for i in range(degree + 1): X[:, i] = t ** i
    return X

# --- Custom Metric Classes ---

class Pelt(BaseEstimator):
    """
    Penalized change point detection using the PELT algorithm.

    This implementation closely follows ruptures.detection.Pelt. Finds change points
    by minimizing a penalized sum of segment costs.
    """
    def __init__(self, model: str = "l2", custom_cost: Optional[BaseCost] = None,
                 min_size: int = 2, jump: int = 5, params: Optional[Dict] = None):
        """Initialize a Pelt instance.

        Args:
            model (str, optional): Segment cost model ("l1", "l2", "rbf", etc.)
                 used if custom_cost is None. Defaults to "l2".
            custom_cost (BaseCost, optional): Custom cost function instance.
                 Overrides `model`. Defaults to None.
            min_size (int, optional): Minimum segment length. Defaults to 2.
            jump (int, optional): Subsample rate for checking breakpoints
                 (one every 'jump' points). Defaults to 5.
            params (dict, optional): Dictionary of parameters for the cost
                 instance if using a standard `model`. Defaults to None.
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            cost_params = params if params is not None else {}
            self.cost = cost_factory(model=model, **cost_params)

        # Ensure min_size respects the cost function's minimum
        self.min_size = max(min_size, getattr(self.cost, 'min_size', 1)) # Use getattr for safety
        self.jump = jump
        self.n_samples = None
        # Note: self.signal is set in fit() by the BaseCost object

    def _seg(self, pen: float) -> Dict[tuple, float]:
        """Computes the segmentation for a given penalty using PELT."""
        partitions = {0: {(0, 0): 0}}
        admissible = []
        ind = list(range(0, self.n_samples, self.jump))
        # Ensure last index and indices >= min_size are included correctly
        if self.n_samples > 0:
             min_idx_start = (self.min_size // self.jump) * self.jump
             ind = [k for k in ind if k >= min_idx_start]
             if self.n_samples not in ind:
                ind.append(self.n_samples)

        for bkp in ind:
            # Add potential previous change point to check
            new_adm_pt = max(0, floor((bkp - self.min_size) / self.jump) * self.jump)
            admissible.append(new_adm_pt)

            subproblems = []
            min_cost = float('inf')

            for t in admissible:
                if t >= bkp: continue # Ensure start is before end
                # Ensure segment t..bkp is long enough
                if bkp - t < self.min_size: continue

                try:
                    cost_t = sum(partitions[t].values())
                    current_segment_cost = self.cost.error(t, bkp)
                    subproblem_cost = cost_t + current_segment_cost + pen
                    subproblems.append((subproblem_cost, t))
                    min_cost = min(min_cost, subproblem_cost)
                except KeyError: # Partition for t doesn't exist
                    continue
                except Exception as e: # Catch cost calculation errors
                    # print(f"Warning: Cost calculation error for segment ({t}, {bkp}): {e}")
                    continue


            if not subproblems: # Handle cases where no valid subproblems found
                 if bkp == 0: continue # Skip if it's the start
                 # Assign infinite cost if no valid prior partition leads here
                 # This shouldn't happen if partitions[0] is set right
                 partitions[bkp] = {(0, bkp): float('inf')}
                 # print(f"Warning: No valid subproblem found for breakpoint {bkp}")
                 continue


            # Find the optimal previous breakpoint t_star
            # Using min_cost calculated during subproblem creation
            best_t = min(subproblems, key=lambda sp: sp[0])[1]

            # Build the optimal partition up to bkp
            partitions[bkp] = partitions[best_t].copy()
            partitions[bkp][(best_t, bkp)] = self.cost.error(best_t, bkp) + pen


            # Pruning step (Rizzo adaptation check)
            admissible = [t for t in admissible if sum(partitions[t].values()) + self.cost.error(t, bkp) <= min_cost + pen]


        best_partition = partitions.get(self.n_samples,{}) # Use .get for safety
        # Remove the initial dummy partition if it exists
        best_partition.pop((0, 0), None)
        return best_partition


    def fit(self, signal: np.ndarray) -> 'Pelt':
        """Fit the cost function to the signal and store signal properties.

        Args:
            signal (array): Signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        if signal.ndim == 1:
            n_samples, = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        self.cost.fit(signal) # Fit the cost function
        return self

    def predict(self, pen: float) -> List[int]:
        """Return the optimal breakpoints for a given penalty.

        Must be called after fit().

        Args:
            pen (float): Penalty value (>0).

        Returns:
            list: Sorted list of breakpoint indices (end points of segments).
        """
        if self.n_samples is None:
             raise ValueError("fit() must be called before predict()")
        partition = self._seg(pen=pen)
        bkps = sorted(e for s, e in partition.keys())
        # Add 0 if it's not implicitly the start of the first segment found
        # if 0 not in {s for s, e in partition.keys()} and bkps:
        #    bkps.insert(0, 0) # Usually PELT returns end points, 0 isn't needed unless no segments found
        # Ensure n_samples is the last breakpoint if signal has length > 0
        if self.n_samples > 0 and (not bkps or bkps[-1] != self.n_samples) :
            #This can happen if the optimal partition is just (0, n_samples)
             if not any(e == self.n_samples for s,e in partition.keys()):
                 #If the only segment is (0, n_samples), keys are just {(0, n_samples): cost}, so bkps is [n_samples]
                 # If partition is empty (error?), ensure n_samples is there
                 if self.n_samples not in bkps:
                    bkps.append(self.n_samples)


        # Return unique sorted list including the end point n_samples
        # The output should be the END indices of the segments.
        # Example: signal[0..10], segments (0,3), (3,7), (7,10) -> bkps=[3, 7, 10]
        final_bkps = sorted(list(set(bkps)))
        # Ensure the final breakpoint is the length of the series, unless empty
        if self.n_samples > 0 and (not final_bkps or final_bkps[-1] != self.n_samples):
             final_bkps.append(self.n_samples)
             final_bkps=sorted(list(set(final_bkps)))

        # Remove 0 if it crept in, PELT typically returns end points > 0
        if 0 in final_bkps:
            final_bkps.remove(0)

        return final_bkps


    def fit_predict(self, signal: np.ndarray, pen: float) -> List[int]:
        """Fit to the signal and return the optimal breakpoints."""
        self.fit(signal)
        return self.predict(pen)


class STLFeatures:
    """
    Calculates time series features based on STL decomposition.

    Uses statsmodels.tsa.seasonal.STL for decomposition and calculates
    features like trend/seasonal strength, linearity, curvature etc.
    """
    def __init__(self, freq: int = 1,
                 seasonal: int = 7,
                 robust: bool = False):
        """Initialize STLFeatures.

        Args:
            freq (int, optional): Frequency of the time series (e.g., 12 for
                monthly data). Mapped to 'period' in statsmodels STL. Defaults to 1.
            seasonal (int, optional): Length of the seasonal smoother. Must be odd.
                Defaults to 7. Passed to statsmodels STL.
            robust (bool, optional): Flag indicating whether to use robust weights
                for the STL decomposition. Defaults to False. Passed to statsmodels STL.
        """
        if freq <= 0:
             raise ValueError("freq (period) must be positive.")
        if seasonal <= 0 or seasonal % 2 == 0:
            raise ValueError("seasonal smoother length must be positive and odd.")

        self.freq = freq
        self.seasonal = seasonal
        self.robust = robust

    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """Calculates STL features for a time series.

        Args:
            x (numpy array): The time series array (1-dimensional).

        Returns:
            dict: A dictionary containing calculated STL features.
                  Returns NaNs for features if decomposition fails or series is too short.
        """
        m = self.freq
        n = len(x)

        # Basic conditions check
        min_len = 2 * m + 1 if m > 1 else 2 # STL needs at least 2 periods
        if n < min_len:
             print(f"Warning: Series too short (len {n} < {min_len}) for STL with period {m}. Returning NaNs.")
             # Return dict with NaNs, structure depends on whether m > 1
             output = {'nperiods': int(m>1), 'seasonal_period': m, 'trend': np.nan, 'spike': np.nan, 'linearity': np.nan, 'curvature': np.nan, 'e_acf1': np.nan, 'e_acf10': np.nan}
             if m > 1: output.update({'seasonal_strength': np.nan, 'peak': np.nan, 'trough': np.nan})
             return output

        trend0 = np.full(n, np.nan)
        seasonal = np.full(n, np.nan)
        remainder = np.full(n, np.nan)

        try:
            if m > 1:
                stlfit = STL(x, period=m, seasonal=self.seasonal, robust=self.robust).fit()
                trend0 = stlfit.trend
                remainder = stlfit.resid
                seasonal = stlfit.seasonal
            else: # Use lowess for trend if non-seasonal
                # lowess requires float type
                trend0 = lowess(x, np.arange(n), frac=0.6, it=2)[:, 1] # Example params, might need tuning
                remainder = x - trend0
                seasonal = np.zeros(n) # No seasonality

        except Exception as e:
            print(f"Warning: STL decomposition failed: {e}. Returning NaNs.")
            # Return dict with NaNs
            output = {'nperiods': int(m>1), 'seasonal_period': m, 'trend': np.nan, 'spike': np.nan, 'linearity': np.nan, 'curvature': np.nan, 'e_acf1': np.nan, 'e_acf10': np.nan}
            if m > 1: output.update({'seasonal_strength': np.nan, 'peak': np.nan, 'trough': np.nan})
            return output

        # Proceed with feature calculation only if decomposition was successful
        varx = np.nanvar(x, ddof=1)
        vardeseason = np.nanvar(x - seasonal, ddof=1)
        vare = np.nanvar(remainder, ddof=1)

        # Trend Strength
        trend_strength = np.nan
        if not np.isclose(varx, 0) and not np.isclose(vardeseason, 0):
            trend_strength = max(0., min(1., 1. - (vare / vardeseason)))

        # Seasonal Strength (only if m > 1)
        seasonal_strength = np.nan
        peak = np.nan
        trough = np.nan
        if m > 1:
            var_seas_rem = np.nanvar(seasonal + remainder, ddof=1)
            if not np.isclose(varx, 0) and not np.isclose(var_seas_rem, 0):
                seasonal_strength = max(0., min(1., 1. - (vare / var_seas_rem)))

            # Peak/Trough calculation needs valid seasonal component
            if not np.all(np.isnan(seasonal)):
                peak_idx = np.nanargmax(seasonal)
                trough_idx = np.nanargmin(seasonal)
                # Ensure indices are valid before modulo
                if not np.isnan(peak_idx): peak = (peak_idx % m) + 1
                if not np.isnan(trough_idx): trough = (trough_idx % m) + 1


        # Spikiness
        d = (remainder - np.nanmean(remainder)) ** 2
        varloo = (vare * (n - 1) - d) / (n - 2) if n > 2 else np.full(n, np.nan)
        spike = np.nanvar(varloo, ddof=1) if n > 2 else np.nan

        # Linearity & Curvature (from trend component)
        linearity = np.nan
        curvature = np.nan
        if not np.all(np.isnan(trend0)) and n >= 3: # Need at least 3 points for quadratic fit
             time = np.arange(n)
             try:
                 # Using poly directly instead of separate function
                 X = np.vstack([time**p for p in range(3)]).T # constant, t, t^2
                 # Ensure X is float
                 X = X.astype(float)
                 # OLS requires float endog too
                 trend0_float = trend0.astype(float)

                 ols_model = OLS(trend0_float, X, missing='drop') # Handle potential NaNs in trend0
                 ols_results = ols_model.fit()
                 coefs = ols_results.params
                 if len(coefs) >= 2: linearity = coefs[1] # Coeff for time
                 if len(coefs) >= 3: curvature = -coefs[2] # Coeff for time^2 (- sign seems conventional here)
             except Exception as e:
                  print(f"Warning: OLS fit for linearity/curvature failed: {e}")


        # ACF features of the remainder
        e_acf1 = np.nan
        e_acf10 = np.nan
        if not np.all(np.isnan(remainder)):
            # Using default nlags=10 here, could make ACF_Features configurable too
            acf_calculator = ACF_Features(nlags=10) # Instantiate with desired nlags
            try:
                acf_result = acf_calculator.get_features(remainder)
                e_acf1 = acf_result.get('x_acf1', np.nan)
                e_acf10 = acf_result.get('x_acf10', np.nan)
            except Exception as e:
                 print(f"Warning: ACF calculation on remainder failed: {e}")

        # Assemble features
        output = {
            'nperiods': int(m > 1),
            'seasonal_period': m,
            'trend': trend_strength,
            'spike': spike,
            'linearity': linearity,
            'curvature': curvature,
            'e_acf1': e_acf1,
            'e_acf10': e_acf10
        }
        if m > 1:
            output['seasonal_strength'] = seasonal_strength
            output['peak'] = peak
            output['trough'] = trough

        return output


class ACF_Features:
    """
    Calculates autocorrelation features using statsmodels.tsa.stattools.acf.
    """
    def __init__(self, nlags: int = 10):
        """Initialize ACF_Features.

        Args:
            nlags (int, optional): The number of lags to include in the ACF calculation.
                                   Defaults to 10. Passed to statsmodels.acf.
        """
        if nlags <= 0:
            raise ValueError("nlags must be positive.")
        self.nlags = nlags

    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """Calculates autocorrelation coefficients.

        Args:
            x (numpy array): The time series array (1-dimensional).

        Returns:
            dict: Contains 'x_acf1' (ACF at lag 1) and 'x_acf10'
                  (sum of ACF at lags 1 to min(nlags, 10)). Returns NaNs on error.
        """
        if len(x) <= max(1, self.nlags): # Need enough points to calculate lags
             print(f"Warning: Series too short (len {len(x)}) for nlags={self.nlags}. Returning NaNs.")
             return {'x_acf1': np.nan, 'x_acf10': np.nan}

        try:
            # Ensure nlags passed to acf is not more than length-1
            effective_nlags = min(self.nlags, len(x) - 1)
            if effective_nlags <= 0: # Should not happen if len(x)>1, but check
                raise ValueError("Effective nlags <= 0")

            acf_values = acf(x, nlags=effective_nlags, fft=True, missing='drop') # Use fft=True, handle NaNs

            # acf_values includes lag 0, which is always 1
            acf1 = acf_values[1] if len(acf_values) > 1 else np.nan

            # Sum lags 1 to min(effective_nlags, 10)
            # Ensure we don't go beyond calculated lags or lag 10
            max_lag_for_sum = min(effective_nlags, 10)
            acf10 = np.sum(acf_values[1 : max_lag_for_sum + 1]) if max_lag_for_sum > 0 else np.nan

            return {'x_acf1': acf1, 'x_acf10': acf10}
        except Exception as e:
            print(f"Warning: ACF calculation failed: {e}. Returning NaNs.")
            return {'x_acf1': np.nan, 'x_acf10': np.nan}


class CrossingPoints:
    """
    Calculates the number of times a time series crosses its median.
    """
    def __init__(self):
        """Initialize CrossingPoints. Takes no parameters."""
        pass # No parameters needed for this calculation

    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """Calculates the number of median crossing points.

        Args:
            x (numpy array): The time series array (1-dimensional).

        Returns:
            dict: Contains 'crossing_points'. Returns NaN if calculation fails.
        """
        try:
             if len(x) < 2: return {'crossing_points': 0.} # Cannot cross with < 2 points

             midline = np.nanmedian(x) # Use nanmedian for robustness
             if np.isnan(midline): return {'crossing_points': np.nan} # Cannot calculate if median is NaN

             # Ensure boolean comparison handles NaNs (they become False)
             ab = x <= midline
             # Handle potential all-NaNs case if not caught by nanmedian
             if np.all(np.isnan(ab)): return {'crossing_points': np.nan}


             # Detect where the condition changes
             p1 = ab[:-1]
             p2 = ab[1:]
             # A cross happens if (True->False) or (False->True) excluding NaNs
             # NaNs in x will propagate to ab, potentially affecting cross calculation
             # We only count transitions between non-NaN adjacent states
             valid_comparison = ~np.isnan(p1) & ~np.isnan(p2)
             cross = ((p1 & ~p2) | (~p1 & p2)) & valid_comparison

             return {'crossing_points': float(np.sum(cross))}
        except Exception as e:
             print(f"Warning: CrossingPoints calculation failed: {e}. Returning NaN.")
             return {'crossing_points': np.nan}


class LinearRegression:
    """
    Performs linear regression of a time series against time (y = B0 + B1*t).

    Uses statsmodels.regression.linear_model.OLS internally.
    """
    def __init__(self):
        """Initializes the LinearRegression model. Takes no parameters."""
        self.coef_: Optional[float] = None # Slope (B1)
        self.intercept_: Optional[float] = None # Intercept (B0)
        self.results_: Optional[RegressionResults] = None # Full OLS results
        self.n_features_in_: int = 1 # Always 1 (time)
        self._time_steps: Optional[np.ndarray] = None
        self._time_series: Optional[np.ndarray] = None

    def fit(self, time_series: np.ndarray) -> 'LinearRegression':
        """Fits the linear regression model y ~ time.

        Args:
            time_series (np.ndarray): 1-dimensional time series data.

        Returns:
            self: The fitted instance.
        """
        # Input validation (simplified)
        y = np.asarray(time_series, dtype=float)
        if y.ndim != 1: raise ValueError("Input time_series must be 1D.")
        if len(y) < 2: raise ValueError("Need at least two points for linear regression.")

        self._time_series = y
        self._time_steps = np.arange(len(y), dtype=float) # Ensure float for OLS

        # Create exogenous variable matrix [constant, time]
        X = sm.add_constant(self._time_steps)

        # Handle potential NaNs in the time series
        valid_idx = ~np.isnan(y) & ~np.isnan(np.sum(X, axis=1)) # Check NaNs in both y and X
        if np.sum(valid_idx) < 2:
             raise ValueError("Not enough non-NaN data points (need at least 2).")

        y_fit = y[valid_idx]
        X_fit = X[valid_idx, :]


        # Perform OLS
        model = sm.OLS(y_fit, X_fit) # Use data without NaNs
        self.results_ = model.fit()

        # Store coefficients
        self.intercept_ = self.results_.params[0]
        self.coef_ = self.results_.params[1]

        return self

    def predict(self, time_steps: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicts values using the fitted linear model.

        Args:
            time_steps (np.ndarray, optional): Time indices to predict for.
                If None, predicts for the original time steps used in fit().

        Returns:
            np.ndarray: Predicted values.
        """
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if time_steps is None:
            X_pred = sm.add_constant(self._time_steps)
        else:
            t_pred = np.asarray(time_steps, dtype=float)
            if t_pred.ndim != 1: raise ValueError("time_steps must be 1D.")
            X_pred = sm.add_constant(t_pred)

        return self.results_.predict(X_pred)

    def score(self) -> Optional[float]:
        """Returns the R-squared ($R^2$) of the fit on the training data.

        Returns:
            float or None: R-squared value, or None if not fitted.
        """
        if self.results_ is None:
            # raise RuntimeError("Model has not been fitted yet. Call fit() first.")
            return None # Return None instead of raising error if not fitted
        return self.results_.rsquared

    def plot_fit(self, figsize=(10, 6)):
        """Generates a plot showing the original data and the fitted line."""
        if self.results_ is None or self._time_series is None or self._time_steps is None:
            raise RuntimeError("Model has not been fitted or data is missing.")

        predicted_values = self.predict() # Predict on original time steps

        plt.figure(figsize=figsize)
        plt.scatter(self._time_steps, self._time_series, label='Original Data Points', marker='o', s=20, alpha=0.7)
        plt.plot(self._time_steps, predicted_values, color='red', linewidth=2, label=f'Fitted Line (Slope={self.coef_:.4f})')
        plt.xlabel("Time Step")
        plt.ylabel("Time Series Value")
        plt.title(f"Linear Regression Fit ($R^2$={self.score():.3f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Prints the detailed summary from statsmodels OLS results."""
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        print(self.results_.summary())


class EntropyPairs:
    """
    Calculates the feature SB_MotifThree_quantile_hh by calling
    a compiled C function. Automatically applies z-score normalization
    to the input time series before calculation.
    """
    def __init__(self):
        """Initialize EntropyPairs C wrapper."""
        if not C_LIB_LOADED or c_motif3_func is None:
            print("Warning: C library/function for EntropyPairs not loaded. Will return NaN.")

    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """
        Applies z-score normalization to the input array 'x' and then
        calls the C function SB_MotifThree_quantile_hh.
        """
        output = {'entropy_pairs': np.nan}
        if not C_LIB_LOADED or c_motif3_func is None: return output

        # --- Input Validation and Conversion ---
        if not isinstance(x, np.ndarray):
            try: x = np.asarray(x, dtype=float)
            except: print("Warning: EntropyPairs input invalid."); return output
        if x.ndim != 1: print("Warning: EntropyPairs input not 1D."); return output
        n = len(x)

        # NaN/Inf Check
        if np.isnan(x).any() or np.isinf(x).any():
            print("Warning: EntropyPairs input contains NaN/Inf. Cannot z-score reliably. Returning NaN.")
            return output

        # --- Z-Score Normalization ---
        x_processed = x 
        if n > 1: # Need at least 2 points for std dev calculation
            std_dev = np.std(x)
            # Check if std dev is meaningfully non-zero to avoid division by zero
            if std_dev > 1e-8:
                try:
                    # Apply z-score normalization (ddof=0 matches np.std default)
                    x_processed = zscore(x, ddof=0)
                except Exception as e_zscore:
                    print(f"Warning: zscore calculation failed for EntropyPairs: {e_zscore}")
                    # Fallback to using original data if zscore fails unexpectedly
                    x_processed = x

        # --- Call C Function ---
        try:
            # Ensure type and contiguity for ctypes using the potentially normalized array
            x_c = np.ascontiguousarray(x_processed, dtype=np.float64)
            # Call C function with the processed data
            result = c_motif3_func(x_c, n)
            output['entropy_pairs'] = float(result) if not np.isnan(result) else np.nan

        except Exception as e:
            print(f"Warning: Call to C function SB_MotifThree_quantile_hh failed: {e}")
            # Output already initialized with NaN

        return output



class SpectralEntropy:
    """Calculates the Spectral Entropy using scipy.signal for PSD."""
    def __init__(self, sf: float, method: str = "welch", nperseg: Optional[int] = None, normalize: bool = False):
        """Initialize SpectralEntropy."""
        if sf <= 0: raise ValueError("Sampling frequency 'sf' must be positive.")
        if method not in ['fft', 'welch']: raise ValueError("Method must be 'fft' or 'welch'.")
        self.sf=sf; self.method=method; self.nperseg=nperseg; self.normalize=normalize

    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """Calculates spectral entropy using scipy.signal PSD methods."""
        output = {'spectral_entropy': np.nan}; axis = -1 # Operate on last axis

        # Basic validation and cleaning
        if not isinstance(x, np.ndarray):
            try: x = np.asarray(x, dtype=float)
            except: print("Warning: SpectralEntropy input invalid. Returning NaN."); return output
        if x.ndim != 1: print("Warning: SpectralEntropy input not 1D. Returning NaN."); return output
        x_clean = x[~np.isnan(x) & ~np.isinf(x)]; n_clean = len(x_clean)
        if n_clean < 2: return output # Need >= 2 points for PSD

        try:
            # Calculate PSD using scipy.signal
            if self.method == "fft":
                freqs, psd = periodogram(x_clean, fs=self.sf, axis=axis)
            elif self.method == "welch":
                seg_len = self.nperseg if self.nperseg is not None else min(256, n_clean) # Use min for default nperseg
                # Ensure nperseg is not larger than series length for welch
                if n_clean < seg_len:
                    freqs, psd = periodogram(x_clean, fs=self.sf, axis=axis)
                else:
                    freqs, psd = welch(x_clean, fs=self.sf, nperseg=seg_len, axis=axis)

            # Check for empty or zero PSD
            if psd is None or psd.size == 0 or np.sum(psd) < 1e-12:
                output['spectral_entropy'] = 0.0 # Entropy of zero signal is zero
                return output

            # Normalize PSD to get probability distribution
            psd_norm = psd / np.sum(psd, axis=axis) # Normalizes along the axis

            # Calculate Shannon entropy using helper
            # Ensure psd_norm is 1D if input x was 1D
            psd_norm_1d = psd_norm.ravel()
            se = -_xlogx_for_entropy(psd_norm_1d, base=2).sum()

            # Apply normalization if requested
            if self.normalize:
                size_psd = psd_norm_1d.size
                if size_psd > 1:
                    se /= log2(size_psd)
                elif size_psd == 1: # Avoid division by log2(1)=0
                    se = 0.0 # Normalized entropy of single point is 0

            output['spectral_entropy'] = se

        except Exception as e:
            print(f"Warning: SpectralEntropy calculation failed: {e}")

        return output


class HighFluctuation:
    """
    Calculates the feature MD_hrv_classic_pnn40 by calling
    a compiled C function. Automatically applies z-score normalization
    to the input time series before calculation.
    """
    def __init__(self):
        """Initialize HighFluctuation C wrapper."""
        if not C_LIB_LOADED or c_pnn40_func is None:
            print("Warning: C library/function for HighFluctuation not loaded. Will return NaN.")
        # Parameters (threshold=40, scale=1000) are assumed inside the C function

    def get_features(self, x: np.ndarray) -> Dict[str, float]:
        """
        Applies z-score normalization to the input array 'x' and then
        calls the C function MD_hrv_classic_pnn40.
        """
        output = {'high_fluctuation': np.nan}
        if not C_LIB_LOADED or c_pnn40_func is None: return output

        # --- Input Validation and Conversion ---
        if not isinstance(x, np.ndarray):
            try: x = np.asarray(x, dtype=float)
            except: print("Warning: HighFluctuation input invalid."); return output
        if x.ndim != 1: print("Warning: HighFluctuation input not 1D."); return output
        n = len(x)

        # NaN/Inf Check (C code might handle, but z-score won't work well with them)
        if np.isnan(x).any() or np.isinf(x).any():
            print("Warning: HighFluctuation input contains NaN/Inf. Cannot z-score reliably. Returning NaN.")
            return output

        # --- Z-Score Normalization ---
        x_processed = x # Default to original if normalization fails or isn't needed
        if n > 1: # Need at least 2 points for std dev calculation
            std_dev = np.std(x)
            # Check if std dev is meaningfully non-zero to avoid division by zero
            if std_dev > 1e-8:
                try:
                    # Apply z-score normalization (ddof=0 matches np.std default)
                    x_processed = zscore(x, ddof=0)
                except Exception as e_zscore:
                    print(f"Warning: zscore calculation failed for HighFluctuation: {e_zscore}")
                    # Fallback to using original data if zscore fails unexpectedly
                    x_processed = x

        # --- Call C Function ---
        try:
            # Ensure type and contiguity for ctypes using the potentially normalized array
            x_c = np.ascontiguousarray(x_processed, dtype=np.float64)
            # Call C function with the processed data
            result = c_pnn40_func(x_c, n)
            output['high_fluctuation'] = float(result) if not np.isnan(result) else np.nan

        except Exception as e:
            print(f"Warning: Call to C function MD_hrv_classic_pnn40 failed: {e}")
            # Output already initialized with NaN

        return output