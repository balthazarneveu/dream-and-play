import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
        return self.data[idx:idx+self.context_length], self.times[idx:idx+self.context_length], self.labels[idx:idx+self.context_length]


def get_dataloader(path=Path("__data"), batch_size=32):
    csv_files = list(path.glob("*.csv"))
    datasets = [PoseDataset(csv_file) for csv_file in csv_files]
    dataset_mixed = ConcatDataset(datasets)
    dataloader = DataLoader(dataset_mixed, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == "__main__":
    dataloader = get_dataloader()
    # Example usage
    for batch_data, batch_time, batch_labels in dataloader:
        print(batch_time, batch_labels)
        break
