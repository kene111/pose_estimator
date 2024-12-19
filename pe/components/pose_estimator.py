# TORCH PACKAGES
import torch
import torch.optim
import torch.nn as nn
import torchvision.transforms as T
# import torch.backends.cudnn as cudnn

# opencv & others
import cv2
import random

import re
import os
import copy
import numpy as np
from operator import itemgetter

from PIL import Image
# import matplotlib.pyplot as plt

from ..config.system_config import PoseEstimatorConfig

class PoseEstimator:
  def __init__(self):
    JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']
    self.JOINTS = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in JOINTS]
    self.POSE_THRESHOLD = 0.6
    self.get_detached = lambda x: copy.deepcopy(x.cpu().detach().numpy())
    self.get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])
    self.model = PoseEstimatorConfig.pe_model
    self.transform = T.Compose([
                       T.Resize((256, 256)),
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                       ])
    self.POSE_PAIRS = [
                        # UPPER BODY
                                      [9, 8],
                                      [8, 7],
                                      [7, 6],

                        # LOWER BODY
                                      [6, 2],
                                      [2, 1],
                                      [1, 0],

                                      [6, 3],
                                      [3, 4],
                                      [4, 5],

                        # ARMS
                                      [7, 12],
                                      [12, 11],
                                      [11, 10],

                                      [7, 13],
                                      [13, 14],
                                      [14, 15]
                        ]

  def load_image(self, image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    return image

  def get_pose_indicators(self, image):
    tr_img = self.transform(image)
    output = self.model(tr_img.unsqueeze(0))
    output = output.squeeze(0)
    return output

  def draw_coor_on_image(self, coor, image_path):
    _, OUT_HEIGHT, OUT_WIDTH = coor.shape
    OUT_SHAPE = (OUT_HEIGHT, OUT_WIDTH)
    pose_layers = self.get_detached(x=coor)
    key_points = list(self.get_keypoints(pose_layers=pose_layers))
    is_joint_plotted = [False for i in range(len(self.JOINTS))]
    image_p = cv2.imread(image_path)
    black_image = np.zeros(image_p.shape) # Blank image
    for pose_pair in self.POSE_PAIRS:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]

        IMG_HEIGHT, IMG_WIDTH, _ =  image_p.shape
        from_x_j, to_x_j = from_x_j * IMG_WIDTH / OUT_SHAPE[0], to_x_j * IMG_WIDTH / OUT_SHAPE[0]
        from_y_j, to_y_j = from_y_j * IMG_HEIGHT / OUT_SHAPE[1], to_y_j * IMG_HEIGHT / OUT_SHAPE[1]

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        if from_thr > self.POSE_THRESHOLD and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(black_image, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True

        if to_thr > self.POSE_THRESHOLD and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(black_image, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        if from_thr > self.POSE_THRESHOLD and to_thr > self.POSE_THRESHOLD:
            # this is a joint connection, plot a line
            cv2.line(black_image, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)
    return black_image

  
  def estimate_pose(self, image_path, save=False):
    img_filepath = None
    image = self.load_image(image_path)
    pose_indi_outputs = self.get_pose_indicators(image)
    img = self.draw_coor_on_image(pose_indi_outputs, image_path)
    if save:
      random_hashname =  str(random.getrandbits(128))
      output_path = f'{PoseEstimatorConfig.output_dir}/{random_hashname}_img.jpg'
      img_saved = cv2.imwrite(output_path, img)

      if img_saved:
        img_filepath = output_path
        
    return img, img_filepath


