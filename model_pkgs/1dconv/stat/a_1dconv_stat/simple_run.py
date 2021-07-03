import os
from os import path
import argparse
from dotenv import load_dotenv

import a_1dconv_stat as execf

load_dotenv(verbose=True)

DATA_PATH = os.environ['DATA_PATH']

def get_args(dataset_name, split_name, temp_dir, sub_folder="audio"):

    data_base_dir = path.join(DATA_PATH, "raw")
    split_base_dir = path.join(DATA_PATH, "splits")

    kfold = False
    
    data_dir = path.join(data_base_dir, dataset_name, sub_folder)
    if "kfold" in split_name:
        kfold = True
    
    split_dir = path.join(split_base_dir, "{}-{}".format(dataset_name, split_name))
    
    ret = [
        "--dataset", dataset_name,
        "--data-folder", data_dir,
        "--train-meta", path.join(split_dir, "train.json"),
        "--test-meta", path.join(split_dir, "test.json"),
        "--temp-folder", temp_dir
    ]

    if kfold:
        ret.append("--kfold")
    else:
        for x in [ "--validation-meta", path.join(split_dir, "val.json") ]:
            ret.append(x)
    return ret

def main(in_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-folder", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--sub-folder", type=str, default="audio")

    pargs, other = parser.parse_known_args(in_args)
    pargs = vars(pargs)

    dataset = pargs['dataset']
    temp_folder = pargs['temp_folder']
    split = pargs['split']
    sub_folder = pargs['sub_folder']

    args = get_args(dataset, split, temp_folder, sub_folder=sub_folder)

    args.extend(other)

    execf.main([
        'train',
        *args
    ])

if __name__ == "__main__":
    main()