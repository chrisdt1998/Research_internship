import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, r'C:\Users\Gebruiker\Documents\GitHub\TimeSformer')

from timesformer.models.vit import TimeSformer


os.system("python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml DATA.PATH_TO_DATA_DIR path_to_your_dataset NUM_GPUS 8 TRAIN.BATCH_SIZE 8")