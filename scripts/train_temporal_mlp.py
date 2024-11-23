from dreamnplay.training.dataset import get_dataloader
from transformers import Trainer, TrainingArguments
import torch

from dreamnplay.training.model import MLPBaseline


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