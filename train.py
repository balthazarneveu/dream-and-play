from dreamnplay.training.dataset import get_dataloader
from transformers import Trainer, TrainingArguments
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

def main():
    model = MLPBaseline()
    train_dataloader, valid_dataloader = get_dataloader()

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        model.train()
        for batch_data, batch_time, batch_labels in train_dataloader:
            optimizer.zero_grad()
            output = model(batch_data)
            loss = loss_fn(output, batch_labels)
            loss.backward()
            accuracy = (output.argmax(dim=1) == batch_labels).float().mean()
            optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()} accuracy: {accuracy}")

        model.eval()
        with torch.no_grad():
            for batch_data, batch_time, batch_labels in valid_dataloader:
                output = model(batch_data)
                loss = loss_fn(output, batch_labels)
                accuracy = (output.argmax(dim=1) == batch_labels).float().mean()
                print(f"Validation loss: {loss.item()} accuracy: {accuracy.item()}")

        torch.save(model.state_dict(), "mlpbaseline_model.pth")

if __name__ == "__main__":
    main()