import torch
from torch import Tensor

from os import path
import wandb
import re

ENTITY = "thasthika"
PROJECT = "mer"

WORKING_DIR = path.dirname(path.dirname(__file__))

def magic_combine(x: Tensor, dim_begin: int, dim_end: int):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)

def __get_model_info(fp):
    _cn = list(filter(lambda x: x.startswith("class"),
               open(fp, mode="r").readlines()))[0]
    x = re.search("class ([^(_V)]+)_V([\d+])", _cn)
    model_version = int(x.group(2))
    model_name = x.group(1)
    return {'name': model_name, 'version': model_version}

def __load_model_class(run, model_version):
    runp = run.split(".")
    if len(runp) > 3:
        runp = runp[:-1]

    model_file = "model_v{}.py".format(model_version)
    run_path = path.join(WORKING_DIR, "models", path.join(*runp), model_file)
    model_info = __get_model_info(run_path)
    ModelClsName = "{}_V{}".format(model_info['name'], model_info['version'])

    runp = ".".join(runp)
    pkg_path = "models.{}.model_v{}".format(runp, model_version)

    print("Loading Model Class {} from {}".format(ModelClsName, pkg_path))

    modelMod = __import__(pkg_path, fromlist=[ModelClsName])
    ModelClass = getattr(modelMod, ModelClsName)

    return (ModelClass, model_info)

def load_model(wandbRun: str, modelName: str, modelVersion: str = "1"):
    api = wandb.Api()
    run = api.run("{}/{}/{}".format(ENTITY, PROJECT, wandbRun))
    (ModelClass, _) = __load_model_class(modelName, modelVersion)

    ckpt_file = None
    for x in run.files():
        if x.name.endswith(".ckpt"):
            ckpt_file = x.download()
    
    if ckpt_file is None:
        raise Exception("Could not find a checkpoint file")

    print(ckpt_file)

    model = ModelClass(**run.config)
    # model.load_state_dict(torch.load(ckpt_file))
    model.eval()

    return model