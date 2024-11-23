import pandas as pd
from torch.utils.data import Dataset, DataLoader
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


if __name__ == "__main__":
    # Paths to your 6 CSV files
    csv_file = "__data/2024-11-22-23-37-29.147.csv"
    # Instantiate the Dataset and DataLoader
    dataset = PoseDataset(csv_file)
    print(dataset[0][1], dataset[1][1])

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # # Example usage
    # for batch_data, batch_labels in dataloader:
    #     print(batch_data, batch_labels)
    #     break
