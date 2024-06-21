import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

# load frm csv
df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test_example.csv")
print(len(df1))
print(len(df2))

# Sample data
data = {
    'Date': ['2010-01-01', '2010-01-02', '2010-01-03', '2010-01-04', '2010-01-05'],
    'store': [0, 0, 0, 1, 2],
    'product': [0, 0, 0, 1, 2],
    'number_sold': [801, 810, 818, 796, 808]
}
df = pd.DataFrame(data)


def preprocess_data(df):
    # take mean of all entries that have same date, and drop 'product' and 'store' columns
    df= df.groupby('Date').mean().reset_index()

    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate day_of_year
    df['day_of_year'] = df['Date'].dt.dayofyear

    # Scale number_sold
    scaler = StandardScaler()
    df['number_sold_scaled'] = scaler.fit_transform(df[['number_sold']])

    # One-hot encode product and store
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[['product', 'store']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['product', 'store']))

    # Combine all features into a single DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    # Drop unnecessary columns
    df = df.drop(columns=['Date', 'number_sold', 'product', 'store'])

    return df



df = preprocess_data(df1)

print(df.head())