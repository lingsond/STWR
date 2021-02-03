import os
import sys
import logging
from pathlib import Path
import pprint
import json
from types import SimpleNamespace

from stwr.stwr_ner import stwr_tag
from stwr.utils import read_config_experiments


if __name__ == "__main__":
    config_file = sys.argv[1]
    configs = read_config_experiments(config_file)
    for config in configs:
        args = SimpleNamespace(**config)
        stwr_tag(args)
