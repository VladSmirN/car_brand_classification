 
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint  
from CarModel import CarModel 
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
np.bool = np.bool_
wandb_logger = WandbLogger(log_model="all")
 

from get_dataloaders import get_dataloaders
from get_transform import get_transform
  
parameters={'batch_size':48,
            'magnitude':8,
            'coarse_dropout':0.05,
            'num_epochs':40,
            'seed':312,
            'max_lr':0.001
            }
 
# Pass the config dictionary when you initialize W&B
run = wandb.init(project="lightning_logs", config=parameters)


path_to_csv='/home/vlad/projects/vipaks/src/car_dataset.csv'
pl.seed_everything(parameters['seed'])
 
training_loader, validation_loader = get_dataloaders(path_to_csv, parameters, get_transform('train', parameters), get_transform('test', parameters))
 
lightning_model = CarModel(parameters, len(training_loader))

checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                      save_weights_only=True,
                                      verbose=False,
                                      save_last=True,
                                      mode='min')

trainer = pl.Trainer(
        max_epochs=parameters['num_epochs'],
        accelerator='cuda',
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        logger=wandb_logger
)
 
trainer.fit(lightning_model, train_dataloaders = training_loader, val_dataloaders = validation_loader)


 
 
 
 