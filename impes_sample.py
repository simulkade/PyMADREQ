from importlib import reload
from pymadreq import *
import pymadreq.coreflood as coreflood
import matplotlib.pyplot as plt
import numpy as np
reload(coreflood)

import json
import os
import sys
from pyfvtool import *

# read the input file
with open('examples/sample.json') as f:
    input_data = json.load(f)

