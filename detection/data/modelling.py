import pandas as pd
import torch
from torch.utils.data import Dataset

N_POSE_FEATURES = 33*3
N_MIN_POSE = 33
ACTION_DURATION = 40

class DataCollatorForPoseMoveDetection(Dataset):
    
    def __init__(self,
                df_input,
                n_min_pose = N_MIN_POSE,
                action_duration = ACTION_DURATION):

        self.df_input = df_input
        self.n_min_pose = n_min_pose
        self.action_duration = action_duration
        
    def __len__(self):
        
        return len(self.df_input) - 1

    def __getitem__(self, idx):

        """
        idx is the idx of an element in the dataset, a number between 0 and len(dataset)
        """

        if idx<=self.n_min_pose:
            idx = self.n_min_pose + torch.randint(0, len(self.df_input)-self.n_min_pose, (1,)).item()

        input = self.df_input.iloc[idx-self.action_duration:idx, :]
        positions = torch.tensor(input['relative_time'].values).float() # time positions
        pose_features = torch.tensor(input.iloc[:, 1:-1].values).float() # pose features, each point with (x,y,p)
        labels = torch.tensor(input['action_code'].values[-1], dtype = torch.long) # action of the last frame
        attention_mask = torch.ones(self.action_duration, dtype = torch.long) # no masking at this point
        
        sample = {"labels": labels,
                  "pose_features": pose_features, 
                  "positions": positions, 
                  "attention_mask": attention_mask}

        return sample

class DataCollatorForPoseMoveDetectionV2(Dataset):
    
    def __init__(self,
                df_input,
                n_min_pose = N_MIN_POSE,
                max_timestamps = ACTION_DURATION):

        self.df_input = df_input
        self.n_min_pose = n_min_pose
        self.max_timestamps = max_timestamps
        
    def __len__(self):
        
        return len(self.df_input) - 1

    def __getitem__(self, idx):

        """
        idx is the idx of an element in the dataset, a number between 0 and len(dataset)
        """

        if idx<=self.n_min_pose:
            idx = self.n_min_pose + torch.randint(0, len(self.df_input)-1-self.n_min_pose, (1,)).item()

        timestamp = self.df_input.iloc[idx, 0]
        input = self.df_input[(self.df_input.relative_time<timestamp) & (self.df_input.relative_time>=timestamp-1)]
        
        n_timestamps = len(input)
        positions = torch.tensor(input['relative_time'].values) # time positions
        positions = torch.cat([positions, 2*torch.ones(self.max_timestamps - n_timestamps)], dim=0).float() # 2 as timestamp, a random choice for padding pose
        
        pose_features = torch.tensor(input.iloc[:, 1:-1].values) # pose features, each point with (x,y,p)
        pose_features = torch.cat([pose_features, torch.zeros(self.max_timestamps - n_timestamps, pose_features.shape[1])], dim = 0).float()

        labels = torch.tensor(input['action_code'].values, dtype = torch.long) # action of all frames
        labels = torch.cat([labels, -100*torch.ones(self.max_timestamps - n_timestamps, dtype = torch.long)], dim=0)

        attention_mask = torch.ones(n_timestamps, dtype = torch.long) # no masking at this point
        attention_mask = torch.cat([attention_mask, torch.zeros(self.max_timestamps - n_timestamps)], dim=0)
        
        sample = {"labels": labels,
                  "pose_features": pose_features, 
                  "positions": positions, 
                  "attention_mask": attention_mask}

        return sample

