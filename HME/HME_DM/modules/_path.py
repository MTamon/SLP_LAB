"""add import path"""

import os
import sys

MODULE_PATH = "/".join(os.path.dirname(__file__).split(os.sep))
sys.path.append(MODULE_PATH)
