import run_kfold

flist = open("./run_list", mode="r").readlines()

for config_file in flist:
    if config_file.endswith(".yaml"):
        print("Running {}...".format(config_file))
        run_kfold.run(config_file)