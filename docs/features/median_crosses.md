## **median_crosses**

Counts the number of times a time-series crosses the median line.  
**Low value:** Means there are few/none oscillations across the time-series.  
**High value:** Means there are frequent oscillations across the time-series.


    
![png](median_crosses_output_5_0.png)
    


##### **No Parameters**

##### **Calculation**

1.  **Median Value:** The median value of the entire time series is calculated.

2.  **Crosses Counting:** Then, for each point Yt and its preceding point Yt−1, if (Yt>median and Yt−1<median) or (Yt<median and Yt−1>median) a crossing is counted.

3.  **Total Count:** The total count of crossings is returned.



##### **Practical Usefulness Examples**

**Process Control:** In manufacturing, if a quality metric frequently crosses its median, it might indicate process instability requiring investigation, even if the average remains acceptable.

**Environmental Monitoring:** Tracking how often a pollutant level crosses its long-term median can highlight periods of increased fluctuation or unusual activity.


