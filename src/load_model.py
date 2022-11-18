from models import LSTMBertModel
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, cfg, args):
        super(Model, self).__init__()
        
        self.cfg = cfg
        self.args = args   

    def forward(self):
        model = LSTMBertModel(self.cfg, self.args)

        return model