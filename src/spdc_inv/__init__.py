# -*- coding: utf-8 -*-
from pathlib import Path
from pkg_resources import get_distribution, DistributionNotFound

import os

try:
    dist_name = "spdc_inv"
    __version__ = get_distribution(dist_name).version

except DistributionNotFound:
    __version__ = 'unknown'

finally:
    del get_distribution, DistributionNotFound

PROJECT_ROOT = Path(Path(__file__).resolve().parents[2])
if str(PROJECT_ROOT).startswith(os.getcwd()):
    PROJECT_ROOT = PROJECT_ROOT.relative_to(os.getcwd())

SRC_ROOT = PROJECT_ROOT.joinpath('src')
PACKAGE_ROOT = SRC_ROOT.joinpath('spdc_inv')
DATA_DIR = PROJECT_ROOT.joinpath('data')
LOGS_DIR = PROJECT_ROOT.joinpath('logs')
RES_DIR = PROJECT_ROOT.joinpath('results')
