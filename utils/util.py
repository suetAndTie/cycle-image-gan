import json
import yaml
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from easydict import EasyDict as edict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return edict(json.load(handle, object_hook=OrderedDict))

def read_yaml(fname):
    with fname.open('rt') as handle:
        return edict(yaml.load(handle))

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=4)

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
