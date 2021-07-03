from posixpath import dirname
import sys
import yaml
from os import path
from dotenv import load_dotenv
import os
import simple_run

load_dotenv(verbose=True)

CONFIG_DIR = path.join(dirname(__file__), 'configs')

def main(config_file):

    config = yaml.load(open(path.join(CONFIG_DIR, config_file), mode="r"), Loader=yaml.FullLoader)

    base_temp_path = os.environ['TEMP_PATH']

    for k in ['temp_folder']:
        if k in config:
            config[k] = path.join(base_temp_path, config[k])
    
    print("Temp Folder: {}".format(config['temp_folder']))

    args = []
    for k in config:
        arg = "--{}".format(k.replace('_', '-'))
        if type(config[k]) is bool:
            if config[k]:
                args.append(arg)
            continue
        args.append(arg)
        args.append(str(config[k]))

    simple_run.main(args)

if __name__ == "__main__":
    config_file = sys.argv[1]    
    main(config_file)