from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

def get_activation_module(activation="relu"):
    _activation = None
    if activation == "relu":
        _activation = nn.ReLU()
    elif activation == "sigmout":
        _activation = nn.Sigmoid()
    elif activation == "tanh":
        _activation = nn.Tanh()
    return _activation

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), batch_normalize=True, dropout=True, dropout_p=0.5, activation="relu"):
        super(Conv2DBlock, self).__init__()

        mod_list = []

        _conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        mod_list.append(("conv1", _conv1))

        _conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        mod_list.append(("conv2", _conv2))
        
        if batch_normalize:
            _bn = nn.BatchNorm2d(out_channels)
            mod_list.append(("bn", _bn))

        if dropout:
            _dropout = nn.Dropout(p=dropout_p)
            mod_list.append(("dropout", _dropout))
        
        _activation = get_activation_module(activation)
        if not _activation is None:
            mod_list.append(("activation", _activation))

        self.net = nn.Sequential(OrderedDict(mod_list))

    def forward(self, x):
        return self.net(x)

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, batch_normalize=True, dropout=True, dropout_p=0.5, activation="relu"):
        super(Conv1DBlock, self).__init__()

        mod_list = []

        _conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        mod_list.append(("conv1", _conv1))

        _conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        mod_list.append(("conv2", _conv2))
        
        if batch_normalize:
            _bn = nn.BatchNorm1d(out_channels)
            mod_list.append(("bn", _bn))

        if dropout:
            _dropout = nn.Dropout(p=dropout_p)
            mod_list.append(("dropout", _dropout))
        
        _activation = get_activation_module(activation)
        if not _activation is None:
            mod_list.append(("activation", _activation))

        self.net = nn.Sequential(OrderedDict(mod_list))

    def forward(self, x):
        return self.net(x)

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, batch_normalize=True, dropout=True, dropout_p=0.5, activation="relu"):
        super(LinearBlock, self).__init__()

        mod_list = []

        _linear = nn.Linear(in_features=in_features, out_features=out_features)
        mod_list.append(("linear", _linear))

        if batch_normalize:
            _bn = nn.BatchNorm1d(out_features)
            mod_list.append(("bn", _bn))

        if dropout:
            _dropout = nn.Dropout(p=dropout_p)
            mod_list.append(("dropout", _dropout))
        
        _activation = get_activation_module(activation)
        if not _activation is None:
            mod_list.append(("activation", _activation))
            
        self.net = nn.Sequential(OrderedDict(mod_list))

    def forward(self, x):
        return self.net(x)