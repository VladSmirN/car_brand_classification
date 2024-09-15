import torch
import PIL
from torch.utils.data import Dataset
car_type_to_class = {'Acura_MDX':0, 'Acura_NSX':1, 'Acura_ILX':2}

class CarDataset(Dataset):
    def __init__(self,paths, car_types, transform):
        self.paths = paths
        self.car_types  = car_types
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = PIL.Image.open(img_path)
        class_ = torch.tensor(car_type_to_class[self.car_types[idx]]) 
        image = self.transform(image)
        return image, class_
