import matplotlib.pyplot as plt
import pandas as pd
import textwrap

data = {
    "Use Case": ["Trend Analysis", "Noise/Complexity", "Seasonality Detection",
                 "Volatility/Outliers", "Model Selection", "Clustering/Classification"],
    "Features": [
        "trend_strength, median_crosses, trend_changes, \nlinear_regression_slope, linear_regression_r2",
        "forecastability, entropy_pairs, fluctuation",
        "ac_relevance, seasonal_strength",
        "window_fluctuation",
        "st_variation, ac_diff_series, complexity",
        "records_concentration, centroid"
    ]
}

df = pd.DataFrame(data)

# Create a figure with adjusted width ratios
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table with adjusted column widths
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='left',
    colWidths=[0.35, 0.65]  # 20% for Use Case, 80% for Features
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(15)

# Adjust row heights and column widths
table.scale(1, 1.5)  # y-scale for better row spacing

# Header color
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Header row
        cell.set_facecolor("#4b8bbe")  # Blue
        cell.set_text_props(color='white', weight='bold')
    cell.set_height(0.15)  # Reduce cell height
    if j == 0:  # Use Case column
        cell.set_width(0.35)  # Narrower
    else:  # Features column
        cell.set_width(0.65)  # Wider

# Adjust cell padding and alignment
for key, cell in table.get_celld().items():
    cell.set_text_props(va='center', ha='left')  # Center vertically, left align
    cell.PAD = 0.1  # Reduce padding

plt.title("Deep Time Series Analysis Features", fontsize=14, weight='bold', pad=20)
plt.savefig("features_table.png", dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()