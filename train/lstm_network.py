import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1):
        super(LSTMNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # The output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Only take the output from the final timetep
        out = out[:, -1, :]
        
        # Pass the output of the LSTM to the output layer
        out = self.fc(out)

        return out
