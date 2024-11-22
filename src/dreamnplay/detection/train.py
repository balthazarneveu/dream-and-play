import torch
import pandas as pd
import json
import glob

from transformers import Trainer, TrainingArguments
from torch.utils.data import ConcatDataset, random_split

from data.modelling import DataCollatorForPoseMoveDetection
from model.movedect import  TransformerForPoseMoveDetection
from utils import compute_metrics, count_parameters

config = json.load(open("config/movedetect.json", "r"))

#----------------- Load the data -----------------#
datafile, batch_size = config["datafile"], config["batch_size"]
dev_ratio, seed = config["dev_ratio"], config["seed"]

files = glob.glob(datafile+'/*.csv')

all_datasets = []
for file in files:
    df_input = pd.read_csv(file)
    all_datasets.append(DataCollatorForPoseMoveDetection(df_input))

my_dataset = ConcatDataset(all_datasets[1:])
# split the dataset 80/20
train_dataset, eval_dataset = random_split(my_dataset, [int((1-dev_ratio)*len(my_dataset)), len(my_dataset) - int((1-dev_ratio)*len(my_dataset))], generator=torch.Generator().manual_seed(seed))   

"""
# in case this take too much time, pre-store all the batches
my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

# upsample the matches data
print("\nUpsampling data...")
all_batches = []
for batch in my_dataloader:
    all_batches.append(batch)

dev_size = int(dev_ratio*len(all_batches))
"""

#----------------- Load the model -----------------#
embed_size, num_layers, num_heads = config["embed_size"], config["num_layers"], config["num_heads"]
model = TransformerForPoseMoveDetection(embed_size=embed_size,
                                        num_layers=num_layers,
                                        heads=num_heads)

count_parameters(model, print_table=True)

#----------------- Train the model -----------------#
output_dir, num_train_epochs, learning_rate, warmup_ratio = config["output_dir"], config["num_train_epochs"], config["learning_rate"], config["warmup_ratio"]
evaluation_strategy, eval_steps, logging_strategy, logging_steps = config["evaluation_strategy"], config["eval_steps"], config["logging_strategy"], config["logging_steps"]
save_strategy, save_steps = config["save_strategy"], config["save_steps"]

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    report_to="tensorboard",
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps, 
    logging_strategy=logging_strategy,
    logging_steps=logging_steps, 
    save_strategy = save_strategy,
    save_steps = save_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()