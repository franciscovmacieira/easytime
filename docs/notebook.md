## Pelt_Num_Breakpoints



This feature detects the number of points where the trend changes.  
**Low value:** The trend has few/none shifting points, and is constant through time.  
**High value:** The trend is constantly shifting, provoking many structural changes.



    
![png](notebook_files/notebook_3_2.png)
    



## STL_Trend_Strength



This feature computes the strength of a trend within the time-series.  
**Low value:** A value close to zero means there are few/none indicators of a trend in the time series.  
**High value:** A value close to one means there are strong signs of the series containing a trend.



    
![png](notebook_files/notebook_3_5.png)
    



## ACF_FirstLag



This feature measures the first 1/e crossing of the auto-correlation function.  
**Low value:** A negative value indicates negative auto-correlation.  
**High value:** A positive value indicates strong auto-correlation.



    
![png](notebook_files/notebook_3_8.png)
    



## LinearRegression_Slope



This feature measures the overall linear trend.  
**Low value:** A negative value means there is a strong downward trend.  
**High value:** A positive value means a strong upward trend.



    
![png](notebook_files/notebook_3_11.png)
    



## LinearRegression_R2



This feature measures the linear fit of a time-series.  
**Low value:** A value close to zero means there is no linear fit.  
**High value:** A value close to one means a high linear fit.



    
![png](notebook_files/notebook_3_14.png)
    

