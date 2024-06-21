from sklearn.model_selection import train_test_split  # for preparing the training and testing data sets. You should get yourself familiar with it.
from sklearn.preprocessing import MinMaxScaler        # Data preprocessing

import torch
import torch.optim as optim
import torch.utils.data as data_util

import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from model import RecurrentNN, RecurrentNNLSTM
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Function to create sequences
def create_sequences(data, history_steps=20, future_steps=5):
    """
    Create specific sequences so that we can feed it into RNN.

    Parameters:
    - data (np.array): The input data.
    - history_steps (int): The number of past steps to consider.
    - future_steps (int): The number of future steps to predict.

    Returns:
    - x (np.array): The input sequences.
    - y (np.array): The output sequences.
    """
    x, y = [], []
    for i in range(len(data) - history_steps - future_steps + 1):
        x.append(data[i:i + history_steps])
        y.append(data[i + history_steps:i + history_steps + future_steps])
    return np.array(x), np.array(y)


def preprocess_data(df):
    """
    Preprocess the input pandas dataframe to prepare it for training. Not all data is used, but can be used if specified in train() function.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - df (pd.DataFrame): The preprocessed DataFrame.
    """
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate day_of_year
    df['day_of_year'] = df['Date'].dt.dayofyear

    # Scale number_sold
    scaler = StandardScaler()
    df['number_sold_scaled'] = scaler.fit_transform(df[['number_sold']])

    # Note that we didn't have the computational capacities to implement this, but we are pretty sure this would improve model performance
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[['product', 'store']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['product', 'store']))

    # Combine all features into a single DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    return df

def train(CurrentModel, input_features=['number_sold_scaled']):
    """
    Train the given model.

    Parameters:
    - CurrentModel (torch.nn.Module): The model class to train.
    - input_features (list): The list of input features.

    Returns:
    - model (torch.nn.Module): The trained model.
    """
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(script_dir, 'train.csv')


    df = pd.read_csv('train.csv', sep=',')

    df = preprocess_data(df)
    data = df[input_features].values.astype("float32")  # Select input features

    # Normalize data
    data = data.reshape(-1, 1).astype("float32")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Parameters
    time_step = 20  # History steps
    future_step = 5  # Future steps

    # Prepare the data
    xtrain, ytrain = create_sequences(data, history_steps=time_step, future_steps=future_step)

    # Convert to tensors
    X = torch.tensor(xtrain, dtype=torch.float32)
    Y = torch.tensor(ytrain, dtype=torch.float32)
    
    # split using train_test_split
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.33, random_state=RANDOM_SEED)

    # Train and validate the imported RNN model
    n_epochs = 20
    mod_epochs = 1 # Model eval epochs

    learning_rate = 0.005

    model = CurrentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    loader = data_util.DataLoader(data_util.TensorDataset(trainX, trainY), shuffle=True, batch_size=8)

    patience = 3
    best_val_rmse = np.inf
    count_epochs_no_improve = 0

    for epoch in range(n_epochs + 1):
        model.train()  # Set the model to training mode
        for X_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch}"):
            y_batch = y_batch.squeeze(-1) if y_batch.dim() == 3 else y_batch  # Ensure y_batch is 2D
            y_pred = model(X_batch)  # One forward pass
            optimizer.zero_grad()  # Clear the gradients
            loss = loss_fn(y_pred, y_batch)  # Compute the loss by this forward pass
            loss.backward()  # Compute the gradients by backpropagation
            optimizer.step()  # The actual step of gradient descent

        if epoch % mod_epochs != 0:
            continue

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y_pred = model(trainX)
            trainY = trainY.squeeze(-1) if trainY.dim() == 3 else trainY  # Ensure trainY is 2D
            train_rmse = torch.sqrt(loss_fn(y_pred, trainY))  # Compute the loss by this forward pass, rmse means root mean squared error
            y_pred = model(valX)
            valY = valY.squeeze(-1) if valY.dim() == 3 else valY  # Ensure testY is 2D
            val_rmse = torch.sqrt(loss_fn(y_pred, valY))
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                count_epochs_no_improve = 0
            else:
                count_epochs_no_improve += 1
                if count_epochs_no_improve == patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch}, train RMSE: {train_rmse}, val RMSE: {val_rmse}")
    return model

if __name__ == "__main__":
    for ModelType in [RecurrentNNLSTM]:
        model = train(ModelType, ['number_sold_scaled'])
        torch.save(model.state_dict(), f'NewModel.pth')