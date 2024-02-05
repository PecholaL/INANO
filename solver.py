""" SRD model + Generator
"""

import yaml
from SRD.model import MAINVC


with open("model.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
SRD = MAINVC(config)
