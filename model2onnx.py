
from CarModel import CarModel 
import torch
import numpy as np



def model2onnx(path_to_pytorch_model, path_to_onnx_model):

    parameters={'batch_size':32,
                'magnitude':6,
                'coarse_dropout':0.05,
                'num_epochs':20,
                'seed':312,
                'max_lr':0.001
                }

    lightning_model = CarModel.load_from_checkpoint(checkpoint_path=path_to_pytorch_model, parameters=parameters, steps_per_epoch=0)
    np_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    torch_input =  torch.from_numpy(np_input)  
    lightning_model.to_onnx(path_to_onnx_model, torch_input, input_names=['input'], output_names=['logit', 'prob'], opset_version=12 )

model2onnx('/home/vlad/projects/vipaks/src/models/model8.ckpt', '/home/vlad/projects/vipaks/src/models/model8.onnx')