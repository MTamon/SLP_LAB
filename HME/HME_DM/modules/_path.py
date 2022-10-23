"""add import path"""

import os
import sys

MODULE_PATH = "/".join(os.path.dirname(__file__).split(os.sep))
sys.path.append(MODULE_PATH)

mem_list = os.listdir(MODULE_PATH)
for m in mem_list:
    PATH = "/".join([MODULE_PATH, m])
    if os.path.isdir(PATH):
        sys.path.append(PATH)
