import torch

import os
from os import path
import re

import sys
import wandb

from models.n1dconv.cat.a.model_v1 import A1DConvCat_V1

ENTITY = "thasthika"
PROJECT = "mer"
WORKING_DIR = path.dirname(__file__)

api = wandb.Api()

device = "cuda" if torch.cuda.is_available() else "cpu"

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

run_name = "1dconv.cat.a"
version = 1
run_id = "kk7mn5lm"

def __get_checkpoint_from_file(run):
    ckpt_file = None
    for f in run.files():
        if f.name.endswith(".ckpt"):
            try:
                ckpt_file = f.download()
                ckpt_file = ckpt_file.name
            except Exception as e:
                print(e)
                ckpt_file = "./{}".format(f.name)
            break
    return ckpt_file

def __get_checkpoint_from_artifact(run):
    pass

def get_model(run_name, run_id, version):
    run = api.run("{}/{}/{}".format(ENTITY, PROJECT, run_id))

    ckpt_file = __get_checkpoint_from_file(run)
    if ckpt_file is None:
        ckpt_file = __get_checkpoint_from_artifact(run)
    if ckpt_file is None:
        print("Warning!! Could not get the checkpoint file")

    config = run.config
    
    # ModelClass, _ = __load_model_class("1dconv.cat.a", 1)
    # model = ModelClass(**config)
    model = A1DConvCat_V1(**config)
    
    if not ckpt_file is None:
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    return model

def main():
    n_args = len(sys.argv)
    if n_args < 5:
        print("exec [run_name] [version] [run_id] [dataset]")
        print("Example: python main.py 1dconv.cat.a 1 kk7mn5lm mer-taffc")
        return

    run_name = sys.argv[1]
    version = sys.argv[2]
    run_id = sys.argv[3]
    dataset = sys.argv[4]

    base_path = "serving"
    dst_dir = path.join(base_path, "models", path.join(*run_name.split(".")))
    os.makedirs(dst_dir, exist_ok=True)

    dst_file = path.join(dst_dir, "model-{}.pt".format(dataset))

    model = get_model(run_name, run_id, version)
    s = torch.jit.script(model)
    print(s)
    torch.jit.save(s, dst_file)

# class TestModel(torch.nn.Module):

#     def __init__(self):
#         super(TestModel, self).__init__()
#         self.l1 = torch.nn.Linear(4, 2)
#         self.l2 = torch.nn.Linear(2, 1)
    
#     def forward(self, x):
#         x = self.l1(x)
#         x = self.l2(x)
#         return x

# def test():
#     model = TestModel()
#     model.eval()
#     s = torch.jit.script(model)
#     torch.jit.save(s, "model.pt")
    

if __name__ == "__main__":
    main()
    # test()

