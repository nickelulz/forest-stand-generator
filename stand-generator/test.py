import sys, os
from stand_dataset import *
import tensorflow as tf
from matplotlib import pyplot

# add root directory
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn import visualize

print('works')
