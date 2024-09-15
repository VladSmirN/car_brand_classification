from torchvision import transforms 
from imgaug import augmenters as iaa
import numpy as np
SIZE = 224 

class CustomTransform:
    def __init__(self, parameters):

        self.aug = iaa.Sequential([
           iaa.Sometimes(0.05,
                          iaa.OneOf([iaa.MotionBlur(k=8, angle=[-20, 20]),
                                     iaa.GaussianBlur(sigma=(0.9, 1.9))])),
            iaa.CoarseDropout(parameters['coarse_dropout']   ),
            
        ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)
    
def get_transform(type_dataset, parameters):

    if type_dataset == 'test':
        return transforms.Compose(
            [
                    transforms.Resize((SIZE, SIZE)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),     
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            ]
        )
    
    if type_dataset == 'train':
        return transforms.Compose(
            [   
                    transforms.Resize((SIZE, SIZE)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=.5, hue=.3),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandAugment(magnitude=parameters['magnitude']),
             
                    CustomTransform(parameters),
                    transforms.ToTensor(),     
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    
    if type_dataset == 'estimate_augmentation':
 
        return transforms.Compose(
            [   
                    transforms.Resize((SIZE, SIZE)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=.5, hue=.3),
                    # transforms.Grayscale(num_output_channels=3),

                    transforms.RandAugment(magnitude=parameters['magnitude']), 
             
                    CustomTransform(parameters),
                    transforms.ToTensor(),      
            ])