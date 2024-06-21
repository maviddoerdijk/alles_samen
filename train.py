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

def train(CurrentModel, input_features=['number_sold']):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'train.csv')

    # with open(file_path, 'r') as f:
    #     # Date,store,product,number_sold
    #     # load to df
    df = pd.read_csv(file_path, sep=',')
    
    df = add_time_features(df)

    data = df[input_features].values
    # plt.plot(data)
    # plt.xlabel('time')
    # plt.ylabel('number_sold')
    # plt.title('Number of sold products over time')
    # plt.savefig('number_sold.png')

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
    trainX = torch.tensor(xtrain, dtype=torch.float32)
    trainY = torch.tensor(ytrain, dtype=torch.float32)

    # Split the data into training and testing sets
    split_idx = int(len(trainX) * 0.67)
    trainX, testX = trainX[:split_idx], trainX[split_idx:]
    trainY, testY = trainY[:split_idx], trainY[split_idx:]

    # Train and validate the imported RNN model
    n_epochs = 100
    mod_epochs = 5  # Model saving epochs

    learning_rate = 0.005

    model = CurrentModel()#.to(device)
    # input_size = 1  # Single feature
    # hidden_size = 50
    # output_size = future_step  # Predicting multiple future steps
    # num_layers = 2
    # model = RecurrentNNLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    # model = RecurrentNNLSTM().to(device)

    loader = data_util.DataLoader(data_util.TensorDataset(trainX, trainY), shuffle=True, batch_size=8)

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
            y_pred = model(testX)
            testY = testY.squeeze(-1) if testY.dim() == 3 else testY  # Ensure testY is 2D
            test_rmse = torch.sqrt(loss_fn(y_pred, testY))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    return model

if __name__ == "__main__":
    for ModelType in [RecurrentNN, RecurrentNNLSTM, RecurrentNNBidirectionalLSTM]:
        for input_features in [['number_sold'], ['number_sold', 'day_of_year']]:
            print(f'Training {ModelType.__name__} with input features {input_features}')
            model = train(ModelType)
            print(f'{model.__class__.__name__}.pth')
            torch.save(model.state_dict(), f'{model.__class__.__name__}.pth')