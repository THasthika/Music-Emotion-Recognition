import yaml
import json
import sys
from os import path
import torchinfo
import torch

def parse_dict_input_data(config):
    ret = {}
    for k in config:
        td = tuple(config[k])
        ret[k] = torch.rand(td)
    return ret

def parse_list_input_data(config):
    if type(config[0]) is int:
        config = [config]

    ## dealing with multiple input arrays
    arr = []
    for a in config:
        td = tuple(a)
        arr.append(torch.rand(td))
    
    if len(arr) == 1:
        return arr[0]
    return arr

def model_info(config_file):
    config = yaml.load(open(config_file, mode="r"), Loader=yaml.FullLoader)

    modelMod = __import__('models.revamped', fromlist=[config['model']['class']])
    ModelClass = getattr(modelMod, config['model']['class'])
    modelConfig = config['model']['config']

    model = ModelClass(**modelConfig)
    print("Model Created...")

    input_data = None
    if 'sample_input' in config['model']:
        si = config['model']['sample_input']
        if type(si) is dict:
            input_data = parse_dict_input_data(si)
        elif type(si) is list:
            input_data = parse_list_input_data(si)

    torchinfo.summary(model, input_data=input_data)

if __name__ == "__main__":
    config_file = sys.argv[1]
    model_info(config_file)