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


if __name__ == '__main__':
    batch_size = 1

    from torch.export import export
    from executorch.exir import to_edge

    model = MLPBaseline()
    model.load_state_dict(torch.load("mlpbaseline_model.pth"))
    pose_features = torch.randn(batch_size, WINDOW_SIZE, POSITION_DIM)
    with torch.no_grad():
        output = model(pose_features)
        aten_dialect = export(model, (pose_features,))
        print(
            f"output.shape: {output.shape}")
        edge_program = to_edge(aten_dialect)
        executorch_program = edge_program.to_executorch()
        with open("mlp_fixed.pte", "wb") as file:
            file.write(executorch_program.buffer)
