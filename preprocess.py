import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

# Sample data
data = {
    'date': ['2010-01-01', '2010-01-02', '2010-01-03', '2010-01-04', '2010-01-05'],
    'store': [0, 0, 0, 1, 2],
    'product': [0, 0, 0, 1, 2],
    'number_sold': [801, 810, 818, 796, 808]
}
df = pd.DataFrame(data)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Calculate day_of_year
df['day_of_year'] = df['date'].dt.dayofyear

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
df = df.drop(columns=['date', 'number_sold', 'product', 'store'])

print(df.head())

# Create sequences for RNN
sequence_length = 3  # Example sequence length
sequences = []
for i in range(len(df) - sequence_length + 1):
    sequence = df.iloc[i:i + sequence_length].values
    sequences.append(sequence)

sequences = np.array(sequences)

# Now, sequences is ready to be used as input for your RNN
print(sequences)