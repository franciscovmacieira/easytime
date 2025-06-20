## **trend_changes**

Detects the number of points where the trend changes.  
**Low value:** The trend has few/none shifting points, and is constant through time.  
**High value:** The trend is constantly shifting, provoking many structural changes.


    
![png](trend_changes_output_5_0.png)
    


##### **Parameters Table**


<style type="text/css">
#T_4ed31 th {
  background-color: #f2f2f2;
  color: black;
  font-weight: bold;
  text-align: left;
  border: 1px solid #ddd;
  padding: 5px;
}
#T_4ed31_row0_col0, #T_4ed31_row1_col0, #T_4ed31_row2_col0, #T_4ed31_row3_col0, #T_4ed31_row4_col0 {
  text-align: left;
  vertical-align: top;
  border: 1px solid #ddd;
  padding: 5px;
  min-width: 100px;
  font-weight: bold;
}
#T_4ed31_row0_col1, #T_4ed31_row1_col1, #T_4ed31_row2_col1, #T_4ed31_row3_col1, #T_4ed31_row4_col1 {
  text-align: left;
  vertical-align: top;
  border: 1px solid #ddd;
  padding: 5px;
  min-width: 60px;
}
#T_4ed31_row0_col2, #T_4ed31_row1_col2, #T_4ed31_row2_col2, #T_4ed31_row3_col2, #T_4ed31_row4_col2 {
  text-align: left;
  vertical-align: top;
  border: 1px solid #ddd;
  padding: 5px;
  min-width: 120px;
  white-space: normal;
  word-wrap: break-word;
}
#T_4ed31_row0_col3, #T_4ed31_row1_col3, #T_4ed31_row2_col3, #T_4ed31_row3_col3, #T_4ed31_row4_col3 {
  text-align: left;
  vertical-align: top;
  border: 1px solid #ddd;
  padding: 5px;
  min-width: 300px;
  max-width: 450px;
  white-space: normal;
  word-wrap: break-word;
}
</style>
<table id="T_4ed31">
  <thead>
    <tr>
      <th id="T_4ed31_level0_col0" class="col_heading level0 col0" >Parameter</th>
      <th id="T_4ed31_level0_col1" class="col_heading level0 col1" >Type</th>
      <th id="T_4ed31_level0_col2" class="col_heading level0 col2" >Default</th>
      <th id="T_4ed31_level0_col3" class="col_heading level0 col3" >Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_4ed31_row0_col0" class="data row0 col0" >model</td>
      <td id="T_4ed31_row0_col1" class="data row0 col1" >str</td>
      <td id="T_4ed31_row0_col2" class="data row0 col2" >'l2'</td>
      <td id="T_4ed31_row0_col3" class="data row0 col3" >Cost function model (e.g., 'l1', 'l2', 'rbf')</td>
    </tr>
    <tr>
      <td id="T_4ed31_row1_col0" class="data row1 col0" >min_size</td>
      <td id="T_4ed31_row1_col1" class="data row1 col1" >int</td>
      <td id="T_4ed31_row1_col2" class="data row1 col2" >2</td>
      <td id="T_4ed31_row1_col3" class="data row1 col3" >Minimum number of samples in a segment.</td>
    </tr>
    <tr>
      <td id="T_4ed31_row2_col0" class="data row2 col0" >jump</td>
      <td id="T_4ed31_row2_col1" class="data row2 col1" >int</td>
      <td id="T_4ed31_row2_col2" class="data row2 col2" >5</td>
      <td id="T_4ed31_row2_col3" class="data row2 col3" >Subsample window for considering change points.</td>
    </tr>
    <tr>
      <td id="T_4ed31_row3_col0" class="data row3 col0" >params</td>
      <td id="T_4ed31_row3_col1" class="data row3 col1" >dict or None</td>
      <td id="T_4ed31_row3_col2" class="data row3 col2" >None</td>
      <td id="T_4ed31_row3_col3" class="data row3 col3" >Additional parameters dictionary for the cost 'model'.</td>
    </tr>
    <tr>
      <td id="T_4ed31_row4_col0" class="data row4 col0" >custom_cost</td>
      <td id="T_4ed31_row4_col1" class="data row4 col1" >BaseCost or None</td>
      <td id="T_4ed31_row4_col2" class="data row4 col2" >None</td>
      <td id="T_4ed31_row4_col3" class="data row4 col3" >Custom cost function (overrides 'model').</td>
    </tr>
  </tbody>
</table>



##### **Calculation**

1.  **Pelt Algorithm (Pruned Exact Linear Time):** The minimum cost for segmenting the series up to a point t is calculated. This is done by considering all possible previous points s. For each s, the known minimum cost to segment up to s is used, and the cost of the current segment (from s to t-1) is added alongside a penalty term. The minimum cost is then the smallest value found among all these possible s points. This cost is computed iteratively for every point in the series.

2. **Breakpoints Counting:** The value returned is the number of detected changepoints (breakpoints) found by backtracking through these optimal choices.



##### **Practical Usefulness Examples**

**Economic Analysis:** Identifying when an economic indicator like GDP growth rate or unemployment changes its trend can signal shifts in the economic cycle, informing policy decisions.

**Marketing Campaign Analysis:** Detecting trend changes in website traffic or conversion rates after launching a marketing campaign can help assess its impact and identify when its effectiveness starts or wanes.


