import sys
import numpy as np
import torch
import cnn.utils as utils
import argparse
import torch.nn as nn
import cnn.genotypes as genotypes
import torch.utils
import torch.backends.cudnn as cudnn
from ltr.cnn.model import NetworkFinal as Network
import cv2 as cv


def numpy_to_torch(a: np.ndarray):
  return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)

seed = 0
gpu = 0
np.random.seed(seed)
torch.cuda.set_device(gpu)
cudnn.benchmark = True
torch.manual_seed(seed)
cudnn.enabled = True
torch.cuda.manual_seed(seed)

arch = 'DARTS'
model_path = ''
videofilepath = ''
genotype = eval("genotypes.%s" % arch)
model = Network(genotype)
model = model.cuda()
utils.load(model, model_path)
model.eval()

cap = cv.VideoCapture(videofilepath)
# cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
# cv.resizeWindow(display_name, 960, 720)

success, frame = cap.read()
while success:
  out = model(numpy_to_torch(frame).cuda())
  center=out.cpu().detach().numpy()
  x,y =center[0,0], center[0,1]
  frame = cv.circle(frame, (x,y), radius=1, color=(0, 0, 255), thickness=-1)
  cv.imshow('Win', frame)
  cv.waitKey(0)
  success, frame = cap.read()
# model.drop_path_prob = args.drop_path_prob

