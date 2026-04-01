from importlib import import_module
import torch.nn as nn
from .layer_groups import flatten_model
from .DATUM import Model as Datum 

class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        
        class para:
            model = 'DATUM'
            n_features=16 if 'n_features' not in kwargs else kwargs['n_features']
            n_blocks=15 if 'n_blocks' not in kwargs else kwargs['n_blocks']
            future_frames=2 if 'future_frames' not in kwargs else kwargs['future_frames']
            past_frames=2 if 'past_frames' not in kwargs else kwargs['past_frames']
            activation='gelu' if 'activation' not in kwargs else kwargs['activation']
            spynet_path=None if 'spynet_path' not in kwargs else kwargs['spynet_path']
            output_full=True if 'output_full' not in kwargs else kwargs['output_full']
        
        self.para = para
        model_name = para.model
        #self.module = import_module('TM_model.{}'.format(model_name))
        self.model = Datum(para) #self.module.Model(para)

    def forward(self, iter_samples):
        iter_samples = (iter_samples['imgs'],iter_samples['tilt'])
        #outputs = self.module.feed(self.model, iter_samples)
        outputs = self.model(iter_samples)
        return {'pred':outputs[0],'pred_noT':outputs[1]}

    def profile(self):
        H, W = self.para.profile_H, self.para.profile_W
        seq_length = self.para.future_frames + self.para.past_frames + 1
        flops, params = self.module.cost_profile(self.model, H, W, seq_length)
        return flops, params
        
    def get_layer_groups(self):    
        return [flatten_model(self.model)]
