import torch
import torch.nn as nn

class RecurrentNNLSTM(nn.Module):
    """
    Eventually found to be the winner!
    """
    def __init__(self, input_size=1, hidden_size=50, output_size=5, num_layers=2):
        super(RecurrentNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out

class RecurrentNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_unit = torch.nn.RNN(input_size=1, hidden_size=10, num_layers=1)
        self.output_unit = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor):
        output,_ = self.rnn_unit(x)
        output = self.output_unit(output[:, -1, :])  # Use the output of the last time step
        return output

class RecurrentNNBidirectionalLSTM(nn.Module):
    def __init__(self):
        super(RecurrentNNBidirectionalLSTM, self).__init__()
        self.rnn_unit = nn.LSTM(input_size=1, hidden_size=10, num_layers=1, bidirectional=True)
        self.output_unit = nn.Linear(20, 5)  # hidden_size * 2 because of bidirectional

    def forward(self, x):
        output, _ = self.rnn_unit(x)
        output = self.output_unit(output[:, -1, :])  # Use the output of the last time step
        return output

class RecurrentNNSeq2Seq(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2):
        super(RecurrentNNSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.encoder(x, (h0, c0))
        out, _ = self.decoder(out)
        out = self.fc(out[:, -1, :])
        return out