from ast import literal_eval as make_tuple
from torchinfo import summary
import argparse

import models

batch_size = 1

# model = Model(**sweep_run.DEFAULT_CONFIG)
# summary(model, input_size=(1, 1, 22050*30), verbose=1)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ModelCatA')
parser.add_argument('--multimodal', type=bool, default=False)
parser.add_argument('--sample_rate', type=int, default=22050)
parser.add_argument('--duration', type=float, default=30.0)
# parser.add_argument('--lyrics_shape', type=str, default="(1, 211)")

args = parser.parse_args()

Model = getattr(models, args.model)
DEFAULT_CONFIG = dict(map(lambda x: (x[0], x[2]), Model.CMDS))

model = Model(init_base=False, **DEFAULT_CONFIG)

if not args.multimodal:
    audio_shape = (batch_size, 1, int(args.sample_rate * args.duration))
    summary(model, input_size=audio_shape, verbose=1)