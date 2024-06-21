from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

df = pd.read_csv("train.csv")
print(len(df))

# plt.savefig('figures/untouched_plot.png')

# $ python extra_plots.py 
#          Date  store  product  number_sold
# 0  2010-01-01      0        0          801
# 1  2010-01-02      0        0          810
# 2  2010-01-03      0        0          818
# 3  2010-01-04      0        0          796
# 4  2010-01-05      0        0          808

# check if every date follows the previous, or the data has gaps / is unsorted
# convert data to datetime
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'].diff().dt.days.ne(1).any()) # True, so there are gaps in the data

# each store, product pair has 3286 days of data, with 6 stores in total, and 9 products in total
#              Date  store  product  number_sold  Date_diff
# 0      2010-01-01      0        0          801        NaN
# 3287   2010-01-01      0        1          851    -3286.0
# 6574   2010-01-01      0        2          893    -3286.0
# 9861   2010-01-01      0        3          802    -3286.0
# 13148  2010-01-01      0        4          938    -3286.0
# ...           ...    ...      ...          ...        ...
# 213655 2010-01-01      6        5          907    -3286.0
# 216942 2010-01-01      6        6          951    -3286.0
# 220229 2010-01-01      6        7          887    -3286.0
# 223516 2010-01-01      6        8          782    -3286.0
# 226803 2010-01-01      6        9          874    -3286.0

# print head of df from idx 3280 to 3290
print(df.iloc[3280:3290])



booleans_to_compare = ['store == 0 and product == 0', 'store == 3 and product == 3', 'store == 6 and product == 6', 'store == 6 and product == 9']

# UNUSED CODE FOR COMPARISON_LARGE
# for store_idx in range(6):
#     for product_idx in range(9):
#         boolean = f'store == {store_idx} and product == {product_idx}'
#         df_filtered = df.query(boolean)
#         df_filtered_mean = df_filtered.groupby('Date').mean().reset_index()
#         plt.plot(df_filtered_mean['number_sold'], label=boolean)

for boolean in booleans_to_compare:
    df_filtered = df.query(boolean)
    df_filtered_mean = df_filtered.groupby('Date').mean().reset_index()
    plt.plot(df_filtered_mean['number_sold'], label=boolean)

plt.title('Some random time series')
plt.legend()
plt.ylabel('number_sold')
plt.xlabel('time (days)')
plt.savefig('figures/comparison.png')
plt.close()

# get average for each date
df_mean = df.groupby('Date').mean().reset_index()


from pandas.plotting import autocorrelation_plot

# Assuming `df_mean` contains the average number of sold products over time
autocorrelation_plot(df_mean['number_sold'])
plt.title('Autocorrelation of Average Number Sold')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
# Calculate the range for x ticks
max_days = df_mean['Date'].max() - df_mean['Date'].min()
max_days = max_days / np.timedelta64(1, 'D')  # Convert to float days
x_ticks = np.arange(0, max_days, 365)
# Set the x ticks
plt.xticks(x_ticks, labels=[str(int(x)) for x in x_ticks])
plt.savefig('figures/autocorrelation_average.png')
plt.close()


from statsmodels.graphics.tsaplots import plot_acf

# Initialize an empty list to store autocorrelation values for each lag
autocorrs = []

# Define the maximum lag 
max_lag = 3285
num_combinations = 0

# Loop through each store and product combination
for store_idx in range(6):
    for product_idx in range(9):
        # Filter the DataFrame for the current store and product
        boolean = f'store == {store_idx} and product == {product_idx}'
        df_filtered = df.query(boolean)
        
        # Check if there are enough data points for meaningful autocorrelation analysis
        if len(df_filtered) > max_lag:
            # Calculate autocorrelation for each lag up to max_lag
            acf_values = pd.Series([df_filtered['number_sold'].autocorr(lag=lag) for lag in range(1, max_lag + 1)])
            
            # Append the autocorrelation values to the list
            autocorrs.append(acf_values)
            num_combinations += 1

# Convert the list of autocorrelation values to a DataFrame for easier manipulation
autocorrs_df = pd.DataFrame(autocorrs)

# Calculate the mean autocorrelation for each lag across all store-product combinations
mean_autocorrs = autocorrs_df.mean()

# Plot the average autocorrelation
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_lag + 1) , mean_autocorrs)# , marker='o', linestyle='-', color='b')
plt.title('Average Autocorrelation across All Store-Product Combinations')
plt.xlabel('Lag')
plt.ylabel('Average Autocorrelation')
plt.grid(True)
plt.xticks(x_ticks, labels=[str(int(x)) for x in x_ticks])

plt.savefig('figures/average_autocorrelation.png')
plt.close()

plt.plot(df_mean['number_sold'])
plt.title('Average number of sold products over time')
plt.ylabel('number_sold')
plt.xlabel('time (days)')
plt.xticks(x_ticks, labels=[str(int(x)) for x in x_ticks])
plt.savefig('figures/average.png')
plt.close()


# use seasonal_decompose to decompose the time series into trend, seasonal, and residual components
result = seasonal_decompose(df_mean['number_sold'], model='additive', period=365)
fig = result.plot()
fig.suptitle('Seasonal decomposition of the average number of sold products', fontsize=8)
plt.savefig('figures/seasonal_decompose.png')
plt.close()



