## **diff_series**

Computes the autocorrelation value of the differenced series.

**Low value:** Means there is no linear relationship between past and current values in the de-trended series.  
**High value:** Means there is a significant linear relationship between past and current values in the de-trended series.


    
![png](diff_series_output_5_0.png)
    


##### **No Parameters**

##### **Calculation**

1.	**First Differencing:** A new time series is created by taking the first differences of the original series, DYt = Yt+1 − Yt for t=1,...,N−1.

2.	**Autocorrelation of Differenced Series:** Then the first 10 autocorrelation coefficients (ρ1,ρ2,...,ρ10) of the differenced series are calculated, using the same method as the ac feature.

3.	**Sum of Squares:** The returned value is calculated as the sum of the squares of these first 10 autocorrelation coefficients.

##### **Practical Usefulness Examples**

**Financial Returns Analysis:** Stock prices are often non-stationary (have a trend/random walk). Analyzing the autocorrelation of their differences (returns) helps identify if there's any remaining predictability after removing the primary random walk component.

**Process Improvement:** If a process output shows a trend, differencing can make it stationary. This feature can then reveal if there are lingering systematic patterns in the rate of change that could be addressed.
