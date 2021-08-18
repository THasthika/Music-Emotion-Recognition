import wandb
import sys

ENTITY = "thasthika"
PROJECT = "mer"

run_id = sys.argv[1]

api = wandb.Api()

run = api.run("{}/{}/{}".format(ENTITY, PROJECT, run_id))

print(run)

for x in run.logged_artifacts():
    if x.type == "model":
        folder_dir = x.download()

        break

# ckpt_file = None
# for x in run.files():
#     print(x)
#     if x.name.endswith(".ckpt"):
#         ckpt_file = x

# print(ckpt_file.)