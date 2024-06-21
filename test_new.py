import numpy as np
import pandas as pd
from model import RecurrentNN
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

history_steps = []
future_steps = []
predicted_steps = []

df = pd.read_csv("test_example.csv")
# df = pd.read_csv("test.csv")
df = df.iloc[:, 1:]
model = RecurrentNN()
model.load_state_dict(torch.load("NNRecurrentModel.pth"))  # load your model here
model.eval()
window_size = 10  # please fill in your own choice: this is the length of history you have to decide
future_steps_size = 5


# split the data set by the combination of `store` and `product``
gb = df.groupby(["store", "product"])
groups = {x: gb.get_group(x) for x in gb.groups}
scores = {}

for key, data in groups.items():
    # By default, we only take the column `number_sold`.
    # Please modify this line if your model takes other columns as input
    X = data.drop(["Date", "store", "product"], axis=1).values  # convert to numpy array
    
    normal_input = X.reshape(-1, 1).astype("float32")
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Load the model here
    scaler.fit(normal_input)
    
    N = X.shape[0]  # total number of testing time steps

    mape_score = []
    start = window_size
    while start + future_steps_size <= N:
        inputs = X[(start - window_size) : start, :]
        targets = X[start : (start + future_steps_size), :]
        
        history_steps.append(inputs)

        # you might need to modify `inputs` before feeding it to your model, e.g., convert it to PyTorch Tensors
        # you might have a different name of the prediction function. Please modify accordingly
        
        #print(f"Inputs for {key}: {inputs.flatten()}")
        
        #normalize inputs

        normal_input = scaler.transform(inputs)
        
        inputs = torch.tensor(normal_input, dtype=torch.float).unsqueeze(0)
        
        #print(f"Normal Inputs for {key}: {inputs.flatten()}")
        
        normalized_predictions = model(inputs).detach().numpy()[0]
        predictions = scaler.inverse_transform(normalized_predictions.reshape(-1,1))
        
        #print(f"Predictions for {key}: {predictions.flatten()}")

        #print(f"Targets for {key}: {targets.flatten()}")

        #print(f"MAPE for {key}: {mean_absolute_percentage_error(targets, predictions)}")
        start += future_steps_size
        # calculate the performance metric
        mape_score.append(mean_absolute_percentage_error(targets, predictions))
        
        future_steps.append(targets)
        predicted_steps.append(predictions)
    print(f"MAPE for {key}: {np.mean(mape_score)}")
    scores[key] = mape_score

# save the performance metrics to file
avg_history = np.mean(history_steps, axis=0)
avg_future = np.mean(future_steps, axis=0)
avg_predicted = np.mean(predicted_steps, axis=0)

plt.plot([i for i in range(window_size)],avg_history, label="History")
plt.plot([i for i in range(window_size-1, window_size+ future_steps_size + 1)],np.append([avg_history[-1]],avg_future), label="Future")
plt.plot([i for i in range(window_size-1, window_size+ future_steps_size + 1)],np.append([avg_history[-1]],avg_predicted), label="Predicted")
plt.legend()
plt.show()

np.savez("score.npz", scores=scores)