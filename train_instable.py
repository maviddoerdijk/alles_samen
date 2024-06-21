# Task: Time Series Forecasting on a Synthetic Data Set
# Data: please see train.csv available on Brightspace
# Specifications:
# You are required to implement a recurrent neural network in PyTorch, which takes as input,
# a recent history of time step  t , e.g., ... ,  t−3 ,  t−2 ,  t−1 ,  t .
# to predict five time step in the future, i.e.,  t+1 ,  t+2 ,  t+3 ,  t+4 ,  t+5 .
# You can use any recurrent NN models taught from the class.
# You could choose the length of the history fed into the model by yourselves.
# The resulting code structure should contain (1) model.py -> the implementation of your own RNN model; (2) train.py -> the training code, which can be executed from the command line by python train.py; (3) requirements.txt that lists the Python packages your are using, including the version information.
# You need to submit your source code and a dumpy file of the best model you ever trained. When handing in the assigment, please put model.py, train.py, requirements.txt, and the model dump file in the same folder named by your group and student IDs. The name must be SUBMISSION__ (e.g., SUBMISSION_NC_PA_24_1_s3610233_s3610234_s3610235). Please see https://pytorch.org/tutorials/beginner/saving_loading_models.html for a tutorial on how to save/load the model.


# DEADLINE: June 21, 2024, 12:00.
# Please deliver your assignment on Brightspace.
# The practical assignment accounts for 30% of the final grade.
# When training your RNN model locally on train.csv, we suggest to use the [Mean Absolute Percentage Error (MAPE)](Mean Absolute Percentage Error) metric to track the performance since we will use this metric to evaluate your model (see below)
# Evaluation criteria:
# Your train.py should be executable - We will contact you in case a bug is encountered. In this case, you will have one chance to fix it, with a penalty of 1 out of 10.
# We will execute your train.py on the training data set train.csv, checking against bugs.
# We will load your best saved model and evaluate it on a testing data set hidden to you.
# Any bugs occur in the evaluation phase will incur a penalty of 1 out of 10.
# The evaluation performance - MAPE - on the testing data will be ranked and the top-5 groups will get a bonus of 2 of 10
import sklearn
from sklearn.model_selection import train_test_split  # for preparing the training and testing data sets. You should get yourself familiar with it.
from sklearn.preprocessing import MinMaxScaler        # Data preprocessing
from sklearn.metrics import accuracy_score            # performance metrics
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data as data_util

import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from model import RecurrentNN, RecurrentNNLSTM, RecurrentNNSeq2Seq, RecurrentNNBidirectionalLSTM

from sklearn.preprocessing import StandardScaler, OneHotEncoder

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Function to create sequences
def create_sequences(data, history_steps=20, future_steps=5):
    x, y = [], []
    for i in range(len(data) - history_steps - future_steps + 1):
        x.append(data[i:i + history_steps])
        y.append(data[i + history_steps:i + history_steps + future_steps])
    return np.array(x), np.array(y)

# Add time-based features to the dataframe
def add_time_features(df):
    df['day_of_year'] = pd.to_datetime(df['Date']).dt.dayofyear
    return df

def preprocess_data(df):
    # take mean of all entries that have same date, and drop 'product' and 'store' columns
    # df= df.groupby('Date').mean().reset_index()

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

    # Drop unnecessary columns
    df = df.drop(columns=['Date', 'number_sold', 'product', 'store'])

    return df


def train(CurrentModel, input_features=['number_sold']):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'train.csv')

    # with open(file_path, 'r') as f:
    #     # Date,store,product,number_sold
    #     # load to df
    df = pd.read_csv(file_path, sep=',')
    
    df = preprocess_data(df)  # Preprocess the data

    data = df[input_features].values.astype("float32")  # Select input features
    # plt.plot(data)
    # plt.xlabel('time')
    # plt.ylabel('number_sold')
    # plt.title('Number of sold products over time')
    # plt.savefig('number_sold.png')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Normalize data
    data = data.reshape(-1, 1)
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

    # Split the data into training and validating sets
    # TODO: split using seed
    split_idx = int(len(X) * 0.67)
    trainX, valX = X[:split_idx], X[split_idx:]
    trainY, valY = Y[:split_idx], Y[split_idx:]
    
    

    # Train and validate the imported RNN model
    n_epochs = 10  # Number of epochs
    mod_epochs = 1  # Model saving epochs

    learning_rate = 0.005

    model = CurrentModel()#.to(device)
  
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
            val_pred = model(valX)
            val_pred = val_pred.squeeze(-1) if val_pred.dim() == 3 else val_pred  # Ensure val_pred is 2D
            val_rmse = np.sqrt(loss_fn(val_pred, valY))
            train_pred = model(trainX)
            train_pred = train_pred.squeeze(-1) if train_pred.dim() == 3 else train_pred  # Ensure train_pred is 2D
            train_rmse = np.sqrt(loss_fn(train_pred, trainY))
            
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
        for input_features in [['number_sold_scaled', 'day_of_year']]: # , 'day_of_year', 'store_0', 'store_1', 'store_2', 'store_3', 'store_4', 'store_5', 'store_6', 'product_0', 'product_1', 'product_2', 'product_3', 'product_4', 'product_5', 'product_6', 'product_7', 'product_8', 'product_9']]: 
            model = train(ModelType, input_features=input_features)
            torch.save(model.state_dict(), f'{model.__class__.__name__}_{len(input_features)}_inputs.pth')