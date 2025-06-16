## **ac**

Computes the autocorrelation of the time-series.

**Low value:** Means the linear relationship between current and past values in the series is very low.  
**High value:** Means the linear relationship between current and past values in the sries is very high, and usually indicates one or more trends.


    
![png](ac_output_5_0.png)
    


##### **No Parameters**

##### **Calculation**

1.	**Lag 1 Autocovariance:** The autocovariance at lag 1 (c1) is calculated.

2.  **Lag 0 Autocovariance:** The autocovariance at lag 0 (c0) is also calculated.

2.	**Autocorrelation Value:** The autocorrelation value that is computed and returned is calculated as: ρ1 = c0 * c1.

##### **Practical Usefulness Examples**

**Demand Forecasting:** If daily sales data has a high positive lag-1 autocorrelation, it means high sales one day are likely followed by high sales the next, useful for short-term inventory adjustments.

**Energy Load Prediction:** Electricity load often shows strong positive lag-1 autocorrelation, as load at one hour is very similar to the previous hour, crucial for grid balancing.

## **diff_series**

Computes the autocorrelation value of the differenced series.

**Low value:** Means there is no linear relationship between past and current values in the de-trended series.  
**High value:** Means there is a significant linear relationship between past and current values in the de-trended series.


    
![png](ac_output_10_0.png)
    


##### **No Parameters**

##### **Calculation**

1.	**First Differencing:** A new time series is created by taking the first differences of the original series, DYt = Yt+1 − Yt for t=1,...,N−1.

2.	**Autocorrelation of Differenced Series:** Then the first 10 autocorrelation coefficients (ρ1,ρ2,...,ρ10) of the differenced series are calculated, using the same method as the ac feature.

3.	**Sum of Squares:** The returned value is calculated as the sum of the squares of these first 10 autocorrelation coefficients.

##### **Practical Usefulness Examples**

**Financial Returns Analysis:** Stock prices are often non-stationary (have a trend/random walk). Analyzing the autocorrelation of their differences (returns) helps identify if there's any remaining predictability after removing the primary random walk component.

**Process Improvement:** If a process output shows a trend, differencing can make it stationary. This feature can then reveal if there are lingering systematic patterns in the rate of change that could be addressed.
