## complexity

Computes the number of structural breaks whithin the series.

**Low value:** A low/null value means the series is relatively constant over time.  
**High value:** A high value means the series is made up of smaller series with different characteristics.


    
![png](complexity_output_5_0.png)
    



<h3>Parameters Table</h3>



<style type="text/css">
#T_a3e2c th {
  background-color: #f2f2f2;
  color: black;
  font-weight: bold;
  text-align: left;
  border: 1px solid #ddd;
  padding: 5px;
}
#T_a3e2c_row0_col0, #T_a3e2c_row1_col0, #T_a3e2c_row2_col0, #T_a3e2c_row3_col0 {
  text-align: left;
  vertical-align: top;
  border: 1px solid #ddd;
  padding: 5px;
  min-width: 100px;
  font-weight: bold;
}
#T_a3e2c_row0_col1, #T_a3e2c_row1_col1, #T_a3e2c_row2_col1, #T_a3e2c_row3_col1 {
  text-align: left;
  vertical-align: top;
  border: 1px solid #ddd;
  padding: 5px;
  min-width: 60px;
}
#T_a3e2c_row0_col2, #T_a3e2c_row1_col2, #T_a3e2c_row2_col2, #T_a3e2c_row3_col2 {
  text-align: left;
  vertical-align: top;
  border: 1px solid #ddd;
  padding: 5px;
  min-width: 120px;
  white-space: normal;
  word-wrap: break-word;
}
#T_a3e2c_row0_col3, #T_a3e2c_row1_col3, #T_a3e2c_row2_col3, #T_a3e2c_row3_col3 {
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
<table id="T_a3e2c">
  <thead>
    <tr>
      <th id="T_a3e2c_level0_col0" class="col_heading level0 col0" >Parameter</th>
      <th id="T_a3e2c_level0_col1" class="col_heading level0 col1" >Type</th>
      <th id="T_a3e2c_level0_col2" class="col_heading level0 col2" >Default</th>
      <th id="T_a3e2c_level0_col3" class="col_heading level0 col3" >Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_a3e2c_row0_col0" class="data row0 col0" >window_size</td>
      <td id="T_a3e2c_row0_col1" class="data row0 col1" >int</td>
      <td id="T_a3e2c_row0_col2" class="data row0 col2" >1</td>
      <td id="T_a3e2c_row0_col3" class="data row0 col3" >Number of data points in each window</td>
    </tr>
    <tr>
      <td id="T_a3e2c_row1_col0" class="data row1 col0" >penalty_value</td>
      <td id="T_a3e2c_row1_col1" class="data row1 col1" >float</td>
      <td id="T_a3e2c_row1_col2" class="data row1 col2" >5</td>
      <td id="T_a3e2c_row1_col3" class="data row1 col3" >Penalty value for the Pelt changing point detection.</td>
    </tr>
    <tr>
      <td id="T_a3e2c_row2_col0" class="data row2 col0" >model</td>
      <td id="T_a3e2c_row2_col1" class="data row2 col1" >str</td>
      <td id="T_a3e2c_row2_col2" class="data row2 col2" >rbf</td>
      <td id="T_a3e2c_row2_col3" class="data row2 col3" >Statistical model to detect changes (e.g. 'rbf', 'l1', 'l2').</td>
    </tr>
    <tr>
      <td id="T_a3e2c_row3_col0" class="data row3 col0" >min_size</td>
      <td id="T_a3e2c_row3_col1" class="data row3 col1" >int</td>
      <td id="T_a3e2c_row3_col2" class="data row3 col2" >2</td>
      <td id="T_a3e2c_row3_col3" class="data row3 col3" >Minimum number of samples between change points.</td>
    </tr>
  </tbody>
</table>


