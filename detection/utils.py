import numpy as np
import pandas as pd
import torch

from prettytable import PrettyTable

def compute_metrics(eval_pred):

    model_output, labels = eval_pred 
    pred, players_embeddings, attention_matrices = model_output 

    # find the most likely predicted player
    pred_classes = np.argmax(pred, axis=1)

    # compute accuracy per label, from 0 to 4
    acc_per_label = []
    for i in range(5):
        idx = labels==i
        acc_per_label.append((pred_classes[idx]==labels[idx]).mean())

    # compute the model top 1 accuracy
    accuracy = (labels==pred_classes).mean()

    outputs = {'accuracy': accuracy,
               'accuracy_nothing': acc_per_label[0],
               'accuracy_activate': acc_per_label[1],
               'accuracy_crouch': acc_per_label[2],
               'accuracy_jump': acc_per_label[3],
               'accuracy_shake': acc_per_label[4]}

    return outputs

def compute_metrics_v2(eval_pred):

    model_output, labels = eval_pred 
    pred, players_embeddings, attention_matrices = model_output 

    # find the most likely predicted player
    pred_classes = np.argmax(pred.reshape(-1, pred.shape[-1]), axis=1)
    labels = labels.reshape(-1)

    # compute accuracy per label, from 0 to 4
    acc_per_label = []
    for i in range(5):
        idx = labels==i
        acc_per_label.append((pred_classes[idx]==labels[idx]).mean())

    # compute the model top 1 accuracy
    accuracy = (labels[labels!=-100]==pred_classes[labels!=-100]).mean()

    outputs = {'accuracy': accuracy,
               'accuracy_nothing': acc_per_label[0],
               'accuracy_activate': acc_per_label[1],
               'accuracy_crouch': acc_per_label[2],
               'accuracy_jump': acc_per_label[3],
               'accuracy_shake': acc_per_label[4]}

    return outputs

def count_parameters(model, print_table = False):

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    
    if print_table:
        print(table)
    
    print(f"Total Trainable Params: {total_params}")
    
    return total_params

def convert_frame_csv(df):
    
    def split_tuple_column(col):
        return col.apply(lambda x: tuple(map(float, x.strip("()").split(','))))

    df_ = pd.DataFrame(df['relative_time'])
    # Transformer le DataFrame
    for col in df.columns[1:-1]:
        # Convertir les chaînes en tuples
        tuples = split_tuple_column(df[col])
        
        # Créer les nouvelles colonnes pour x, y, z, p
        df_[f'{col}_x'] = tuples.apply(lambda t: t[0])
        df_[f'{col}_y'] = tuples.apply(lambda t: t[1])
        df_[f'{col}_p'] = tuples.apply(lambda t: t[3])

    df_ = pd.concat([df_, df['action']], axis=1)

    return df_
