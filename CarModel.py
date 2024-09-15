import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision
import numpy as np

class CarModel(pl.LightningModule):
    def __init__(self, parameters, steps_per_epoch  ):
        super(CarModel, self).__init__()
        self.max_lr = parameters['max_lr'] 
        self.epochs = parameters['num_epochs'] 
 

        self.steps_per_epoch = steps_per_epoch
        self.criterion = torch.nn.CrossEntropyLoss()
  
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", average='micro', num_classes=3)
        self.valid_precision = torchmetrics.classification.Precision(task="multiclass", average='macro',num_classes=3)
        self.valid_recall = torchmetrics.classification.Recall(task="multiclass",average='macro', num_classes=3)

        self.model= torchvision.models.convnext_small(pretrained=True) 
        self.model.classifier[2] = torch.nn.Identity() 
        
        self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.2),
                    torch.nn.Linear(768, 256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.2),
                    torch.nn.Linear(256, 3))

    def forward(self, images):
        embeddings = self.model(images) 
        logit = self.classifier(embeddings)
        prob = torch.nn.functional.softmax(logit, dim=1).cuda()
        return  logit, prob 
    
    def configure_optimizers(self):

        self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.classifier.parameters()}
            ])

        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                                epochs = self.epochs, 
                                                                steps_per_epoch = self.steps_per_epoch,
                                                                max_lr = self.max_lr)

        scheduler = {'scheduler': scheduler, 'interval': 'step'}
 
        return [self.optimizer], [scheduler]  
  
    
    def validation_step(self, batch, batch_idx):
        images, class_   = batch[0], batch[1] 
        logit, prob = self.forward(images) 
        predict = torch.argmax(prob, dim=1)

        loss  = self.criterion(logit, class_)

        self.valid_acc(predict, class_)
        self.valid_precision(predict, class_)
        self.valid_recall(predict, class_)

        self.log('valid_loss', loss.cpu().detach())
    
        return loss

    def training_step(self, batch, batch_idx):
        images, class_   = batch[0], batch[1]
        logit, prob = self.forward(images) 

        loss  = self.criterion(logit, class_)
   
        self.log('train_loss', loss.cpu().detach())
        self.log('lr', self.optimizer.param_groups[0]['lr'])

        return loss
    
    def on_validation_epoch_end(self):
    
        self.log('valid_acc', self.valid_acc.compute())
        self.log('valid_precision', self.valid_precision.compute())
        self.log('valid_recall', self.valid_recall.compute())    
 
        self.valid_acc.reset()
        self.valid_precision.reset()
        self.valid_recall.reset()

 

 