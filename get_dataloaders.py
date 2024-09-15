import os
import torch
import PIL
import pandas as pd
 
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from CarDataset import CarDataset
 
def get_dataloaders(path_to_csv, parameters, train_transform, test_transform):

    df_dataset = pd.read_csv(path_to_csv)
    df_train, df_valid =  train_test_split(df_dataset, test_size=0.3, stratify=df_dataset['car_type'].values)
 
    car_type_to_weight=dict(1/df_train['car_type'].value_counts())
    weights = df_train['car_type'].apply(lambda car_type: car_type_to_weight[car_type]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dataset = CarDataset(df_train['path'].values, df_train['car_type'].values, transform=train_transform)
    valid_dataset = CarDataset(df_valid['path'].values, df_valid['car_type'].values, transform=test_transform)

    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters['batch_size'], sampler=sampler, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    return training_loader, validation_loader

 