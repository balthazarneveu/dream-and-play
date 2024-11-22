import pandas as pd
import torch
from torch.utils.data import Dataset

N_POSE_FEATURES = 33*3
N_MIN_POSE = 39*2
ACTION_DURATION = 39*2

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

