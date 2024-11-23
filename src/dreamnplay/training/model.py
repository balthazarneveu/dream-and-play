import torch

WINDOW_SIZE = 6
POSITION_DIM = 132
NUM_CLASSES = 3


class MLPBaseline(torch.nn.Module):
    def __init__(self, input_dim=POSITION_DIM*WINDOW_SIZE, output_dim=NUM_CLASSES):
        super(MLPBaseline, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_dim)

    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x