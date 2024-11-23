import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch
from pathlib import Path
import ast


class PoseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.labels = []
        self.context_length = 6
        df = pd.read_csv(csv_file)
        self.labels.extend(df['action'].tolist())
        features = df.drop(columns=['relative_time', 'action'])
        for _, row in features.iterrows():
            parsed_row = []
            for col in features.columns:
                parsed_row.extend(ast.literal_eval(row[col]))
            self.data.append(parsed_row)
        self.times = df['relative_time'].tolist()

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        # output sequence of length context_length
        # -> position (x, y, z, v) , timestep, label
        positions = self.data[idx:idx+self.context_length]
        timesteps = self.times[idx:idx+self.context_length]
        labels = self.labels[idx+self.context_length-1]
        return torch.tensor(positions), torch.tensor(timesteps), labels


def get_dataloader(path=Path("__data"), batch_size=32):
    rng = torch.Generator().manual_seed(42)
    csv_files = list(path.glob("*.csv"))
    datasets = [PoseDataset(csv_file) for csv_file in csv_files]
    dataset_mixed = ConcatDataset(datasets)
    train_set, valid_set = random_split(
        dataset_mixed, [0.8, 0.2], generator=rng)
    # print(datasets[0][0][1])
    # print(dataset_mixed[0][1])
    # print(train_set[0][1])
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=True)
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    train_dataloader, valid_dataloader = get_dataloader()
    # Example usage
    for batch_data, batch_time, batch_labels in train_dataloader:
        print(batch_data.shape, batch_time.shape)
        print(batch_time[0], batch_labels)
        break
