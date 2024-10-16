""" solver for training Generator
"""

import yaml
from Generator import Generator

g = Generator()

with open("model.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
