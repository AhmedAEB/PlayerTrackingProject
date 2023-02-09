import sys
from os import path

# add "yolo" directory to PATH (required by YOLO's torch serialized model checkpoint)
sys.path.append(path.dirname(__file__))