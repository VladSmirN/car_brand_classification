import onnxruntime as ort
import numpy as np
import os
import glob 
from get_transform import get_transform
from CarDataset import CarDataset
import pandas as pd


class_to_car_type = {0:'Acura_MDX', 1:'Acura_NSX', 2:'Acura_ILX'}
 
def predict(path_folder, path_to_model):
    paths = glob.glob(os.path.join(path_folder, '*.*')) 
    dataset = CarDataset(paths, ['Acura_MDX']*len(paths), transform=get_transform('test',{}))

    ort_sess = ort.InferenceSession(path_to_model)
    results = []
    for i,(image,_) in enumerate(iter(dataset)):
 
        image = np.expand_dims(image.numpy() , axis=0)  
        _ ,probe = ort_sess.run(['logit', 'prob'],{'input':image})
        class_ = np.argmax(probe)
        
        car_type = class_to_car_type[class_]
        results += [car_type]
    return pd.DataFrame(results,columns=['car_type'])


 