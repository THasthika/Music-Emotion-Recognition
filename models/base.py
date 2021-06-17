from os import path
from collections import OrderedDict

import torch.nn as nn

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from utils.common import Conv2DBlock, Conv1DBlock, LinearBlock

class BaseModel(pl.LightningModule):

    def __init__(self,
                batch_size=32,
                num_workers=4,
                dataset_class=None,
                dataset_class_args={
                    'sample_rate': 22050,
                    'max_duration': 30,
                    'label_type': 'categorical',
                    'audio_dir': ''
                },
                split_dir=None):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_class = dataset_class
        self.dataset_class_args = dataset_class_args
        self.split_dir = split_dir

    def create_conv_network(self, config, adaptive_units, adaptive_pooling_type='avg', conv_type='1d'):
        ## config structure
        ## -> either simple a list of numbers (for channels)
        ## -> either list of dicts with (channels, kernel_size, strid, batch_normalize, dropout, dropout_p, activation)
        ARG_CHANNELS = 'channels'

        ARG_IN_CHANNELS = 'in_channels'
        ARG_OUT_CHANNELS = 'out_channels'
        ARG_KERNEL_SIZE = 'kernel_size'
        ARG_STRIDE = 'stride'
        ARG_BATCH_NORMALIZE = 'batch_normalize'
        ARG_DROPOUT = 'dropout'
        ARG_DROPOUT_P = 'dropout_p'
        ARG_ACTIVATION = 'activation'

        ConvBlock = Conv1DBlock
        if conv_type == '1d':
            ConvBlock = Conv1DBlock
        elif conv_type == '2d':
            ConvBlock = Conv2DBlock


        config = list(map(lambda x: { ARG_CHANNELS: x } if type(x) is int or (type(x) is tuple and type(x[0]) is int) else x, config))
        layer_list = list()
        for i in range(len(config) - 1):
            args = {
                ARG_IN_CHANNELS: config[i][ARG_CHANNELS],
                ARG_OUT_CHANNELS: config[i+1][ARG_CHANNELS],
            }
            for x in [ARG_KERNEL_SIZE, ARG_STRIDE, ARG_BATCH_NORMALIZE, ARG_DROPOUT, ARG_DROPOUT_P, ARG_ACTIVATION]:
                if x in config:
                    args[x] = config[x]
            layer_list.append(('conv{}'.format(i+1), ConvBlock(**args)))

        if adaptive_pooling_type == 'avg':
            layer_list.append(
                ('adaptive_layer', nn.AdaptiveAvgPool1d(adaptive_units))
            )
        elif adaptive_pooling_type == 'max':
            layer_list.append(
                ('adaptive_layer', nn.AdaptiveMaxPool1d(adaptive_units))
            )
        
        return nn.Sequential(OrderedDict(layer_list))

    def create_linear_network(self, config):

        ARG_FEATURES = 'features'

        ARG_IN_FEATURES = 'in_features'
        ARG_OUT_FEATURES = 'out_features'
        ARG_BATCH_NORMALIZE = 'batch_normalize'
        ARG_DROPOUT = 'dropout'
        ARG_DROPOUT_P = 'dropout_p'
        ARG_ACTIVATION = 'activation'

        config = list(map(lambda x: { ARG_FEATURES: x } if type(x) is int or (type(x) is tuple and type(x[0]) is int) else x, config))
        layer_list = list()
        for i in range(len(config) - 2):
            args = {
                ARG_IN_FEATURES: config[i][ARG_FEATURES],
                ARG_OUT_FEATURES: config[i+1][ARG_FEATURES],
            }
            for x in [ARG_BATCH_NORMALIZE, ARG_DROPOUT, ARG_DROPOUT_P, ARG_ACTIVATION]:
                if x in config:
                    args[x] = config[x]
            layer_list.append(('linear{}'.format(i+1), LinearBlock(**args)))
        
        return nn.Sequential(OrderedDict(layer_list))

    def set_model_parameter(self, config, config_keys, default):
        if not (type(config_keys) is list or type(config_keys) is tuple):
            config_keys = [config_keys]
        exists = True
        temp = []
        for k in config_keys:
            if not k in config:
                exists = False
                break
            temp.append(config[k])
        if not exists:
            return default
        if len(temp) == 1:
            return temp[0]
        else:
            return temp

    # def prepare_data(self):

    #     if self.dataset_class is None or self.split_dir is None:
    #         return

    #     split_dir = self.split_dir
    #     DSClass = self.dataset_class
    #     additional_args = self.dataset_class_args

    #     train_meta_file = path.join(split_dir, "train.json")
    #     val_meta_file = path.join(split_dir, "val.json")
    #     test_meta_file = path.join(split_dir, "test.json")

    #     has_val = False
    #     has_test = False
    #     if path.exists(val_meta_file):
    #         has_val = True
    #     if path.exists(test_meta_file):
    #         has_test = True

    #     self.train_ds = None
    #     self.val_ds = None
    #     self.test_ds = None

    #     self.train_ds = DSClass(
    #         meta_file=train_meta_file,
    #         **additional_args)

    #     if has_val:
    #         self.val_ds = DSClass(
    #             meta_file=val_meta_file,
    #             **additional_args)

    #     if has_test:
    #         self.test_ds = DSClass(
    #             meta_file=test_meta_file,
    #             **additional_args)

    # def train_dataloader(self):
    #     if self.test_ds is None: return None
    #     return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     if self.val_ds is None: return None
    #     return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     if self.test_ds is None: return None
    #     return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)