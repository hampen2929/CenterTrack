# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # import 

import sys
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
import time
import cv2
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import math

base_dir = '/workspace/CenterTrack/'
centertrack_path = osp.join(base_dir, 'src', 'lib')
sys.path.insert(0, centertrack_path)

from detector import Detector
from opts import opts

np.set_printoptions(suppress=True, precision=3)


# # function

def queue(arr, arr_add):
    dst = np.vstack([arr, arr_add])
    dst = np.delete(dst, obj=0, axis=0)
    return dst


def gen_nan_mat(dim, data_num=None):
    if data_num is None:
        mat = np.zeros(dim)
    else:
        mat = np.zeros((data_num, dim))
    mat[:] = np.nan
    return mat


# # class

class Data(metaclass=ABCMeta):
    '''
    0: num, 1: frame_index, 2: class_id,
    3: x_ct, 4: y_ct, 5: xmin, 6: ymin, 7: xmax, 8: ymax, 
    9: x_ct_bbox, 10: y_ct_bbox, 
    11: tracking_id, 12: score, 13: x_trk, 14: y_trk, 15: age, 16: active
    '''
    value_dim = 17
    
    def __init__(self, data_num=15):
        self._set_values(data_num)
        
    def _set_values(self, data_num):
        self._values = gen_nan_mat(self.value_dim, data_num)
        
    def update(self, in_values):
        self._in_values = in_values
        self._validate()
        self._process()
        self._set()
    
    def _validate(self):
        if len(self._in_values)==0:
            self._in_values = gen_nan_mat(self.value_dim, 1)
        else:
            self._validate_class_id()
        
    def _validate_class_id(self):
        if not (self._in_values[:, 2]!=self.class_id).sum()==0:
            msg = 'Not {} data({}) is included {}'.format(self.class_name, 
                                                          self.class_id, 
                                                          self._in_values[:, 2])
            raise ValueError(msg)
    
    @abstractmethod
    def _process(self):
        pass


class BallData():
    '''
    0: num, 1: frame_index, 2: class_id,
    3: x_ct, 4: y_ct, 5: xmin, 6: ymin, 7: xmax, 8: ymax, 
    9: x_ct_bbox, 10: y_ct_bbox, 
    11: tracking_id, 12: score, 13: x_trk, 14: y_trk, 15: age, 16: active
    '''
    class_id = 2
    class_name = 'ball'
    value_dim = 17
    
    def __init__(self, data_num=15):
        self._set_values(data_num)
        
    def _set_values(self, data_num):
        self._values = gen_nan_mat(self.value_dim, data_num)
        
    def update(self, in_values):
        self._in_values = in_values
        self._validate()
        self._process()
        self._set()
    
    # validate
    def _validate(self):
        if len(self._in_values)==0:
            self._in_values = gen_nan_mat(self.value_dim, 1)
        else:
            self._validate_class_id()
        
    def _validate_class_id(self):
        if not (self._in_values[:, 2]!=self.class_id).sum()==0:
            msg = 'Not {} data({}) is included {}'.format(self.class_name, 
                                                          self.class_id, 
                                                          self._in_values[:, 2])
            raise ValueError(msg)

    # process
    def _process(self):
        self._filter_highest_score()
        # self.__low_pass_filter()
        self._add_value()

    def _filter_highest_score(self):
        self._value = self._in_values[np.argmax(self._in_values[:, 12])]
    
    def _low_pass_filter(self, k=0.1):
        self._value = k * self._values[-1] + (1 - k) * self._value
        
    def _add_value(self):
        self._values = queue(self._values, self._value)

    # set
    def _set(self):
        self._set_pos()
        self._set_bbox()
    
    def _set_pos(self):
        self._pos = self._value[[9, 10]]
        self._poses = self._values[:, [9, 10]]
        
    def _set_bbox(self):
        self._bbox = self._value[[5, 6, 7, 8]]
        self._bboxes = self._values[:, [5, 6, 7, 8]]
    
    # property
    @property
    def value(self):
        return self._value
    
    @property
    def values(self):
        return self._values
    
    @property
    def center_pos(self):
        return self._pos
    
    @property
    def center_poses(self):
        return self._poses
    
    @property
    def bbox(self):
        return self._bbox
    
    @property
    def bboxes(self):
        return self._bboxes    
    
    # visualize
    def draw(self, frame, trajectory=True):
        if trajectory:
            for pos in self._poses:
                if np.sum(~np.isnan(pos))!=0:
                    cv2.circle(frame, tuple(pos.astype(int)), 6, (0, 255, 255), thickness=-1)
        if np.sum(~np.isnan(self._pos))!=0:
            cv2.circle(frame, tuple(self._pos.astype(int)), 8, (0, 0, 255), thickness=-1)


class CourtData():
    '''
    0: num, 
    1: frame_index
    2: class_id,
    3: x_ct
    4: y_ct
    5: xmin
    6: ymin
    7: xmax
    8: ymax, 
    9: x_ct_bbox
    10: y_ct_bbox, 
    11: tracking_id
    12: score
    13: x_trk
    14: y_trk
    15: age
    16: active
    '''
    class_id = 3
    class_name = 'court'
    value_dim = 9
    
    def __init__(self, 
                 data_num=15, 
                 frame_width=1280, 
                 frame_height=720):
        self._set_values(data_num)
        self._frame_center_pos = self._set_frame_center_pos(frame_width, frame_height)
        
    def _set_values(self, data_num):
        self._poses = gen_nan_mat(self.value_dim, data_num)
    
    def _set_frame_center_pos(self, frame_width, frame_height):
        return (int(frame_width / 2), int(frame_height / 2))
        
    def update(self, values):
        values = self._validate(values)
        self._process(values)
        # self._set(values)
    
    # validate
    def _validate(self, values):
        if len(values)==0:
            values = gen_nan_mat(self.value_dim, 1)
        else:
            self._validate_class_id(values)
        return values
        
    def _validate_class_id(self, values):
        if not (values[:, 2]!=self.class_id).sum()==0:
            msg = 'Not {} data({}) is included {}'.format(self.class_name, 
                                                          self.class_id, 
                                                          values[:, 2])
            raise ValueError(msg)

    # process
    def _process(self, values):
        values = self._filter_values(values)
        self._pos = self._get_edge_pos(values)
        self._add_value(self._pos)

    def _filter_values(self, values):
        """
        2つ以上存在するedgeのフィルター
        """
        quadrants = []
        for v in values:
            score = v[12]
            num = v[0]
            x, y = self._get_polar_coordinates(v)
            theta = self._get_theta(x, y)
            _, quadrant = self._get_edge_type(theta)
            quadrants.append(quadrant)
        quadrants = np.array(quadrants)
        values = np.insert(values, values.shape[1], quadrants, axis=1)
        return values
        
    def _get_edge_pos(self, values):
        """
        0: frame_index
        1: x_1
        2: y_1
        3: x_2
        4: y_2
        5: x_3
        6: y_3
        7: x_4
        8: y_4
        """
        edge_pos = np.zeros(9)
        edge_pos[:] = np.nan
        for v in values:
            edge_pos[0] = v[1]
            quadrant = v[17]
            if quadrant==1:
                edge_pos[1] = int(v[9])
                edge_pos[2] = int(v[10])
            elif quadrant==2:
                edge_pos[3] = int(v[9])
                edge_pos[4] = int(v[10])
            elif quadrant==3:
                edge_pos[5] = int(v[9])
                edge_pos[6] = int(v[10])
            elif quadrant==4:
                edge_pos[7] = int(v[9])
                edge_pos[8] = int(v[10])
            else:
                raise ValueError
        return edge_pos
            
    
    def _get_polar_coordinates(self, v):
        x = v[9] - self._frame_center_pos[0]
        y = v[10] - self._frame_center_pos[1]
        return x, y

    def _get_theta(self, x, y):
        return math.degrees(math.atan2(y, x))
    
    def _get_edge_type(self, theta):
        if 0 <= theta <= 90:
            edge_name = 'doubles_upper_right'
            quadrant = 1
        elif 90 < theta <= 180:
            edge_name = 'doubles_upper_left'
            quadrant = 2
        elif -180 < theta <= -90:
            edge_name = 'doubles_lower_left'
            quadrant = 3
        elif -90 <= theta < 0:
            edge_name = 'doubles_lower_right'
            quadrant = 4
        else:
            raise ValueError
        return edge_name, quadrant
            
    def _add_value(self, value):
        self._poses = queue(self._poses, value)
        
    @property
    def pos(self):
        return self._pos
    
    @property
    def poses(self):
        return self._poses
    
    # visualize
    def draw(self, frame):
        if np.sum(~np.isnan(self._pos))!=0:
            pts = court_data.pos[1:9].reshape((-1, 1, 2)).astype(int)
            cv2.polylines(frame,[pts],True,(255,0,255),thickness=2)
            # 塗りつぶし
            # cv2.fillConvexPoly(frame, pts, (0, 0, 0))

# # model

TASK = 'tracking' # or 'tracking,multi_pose' for pose tracking and 'tracking,ddd' for monocular 3d tracking
MODEL_PATH = osp.join(base_dir, 'models', 'tennis3_11.pth')
# MODEL_PATH = osp.join(base_dir, 'models', 'mot17_fulltrain.pth')

opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
opt.track_thresh=0.4

# # path

movie_path = '../videos/match_01_part.mp4'

# # main

# +
ball_data = BallData(data_num=30)
court_data = CourtData(data_num=30)

pos_values = np.zeros((15, 2))
pos_values[:, :] = np.nan

skip = 1
detector = Detector(opt)
balls = np.array
frame_index = 1
cap = cv2.VideoCapture(movie_path)

datas = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    if frame_index==1:
        input_meta = {'pre_dets': []}
        _ = detector.run(frame, input_meta)
        frame_index += 1
        continue
    
    if frame_index % skip!=0:
        frame_index += 1
        continue
    
    infer_start_time = time.time()
    results = detector.run(frame)['results']
    infer_end_time = time.time()
    
    parse_start_time = time.time()
    res_arr = np.zeros((len(results), 17))
    for num, result in enumerate(results):
        res_val = result.values()
        score, class_id, ct, tracking, bbox, tracking_id, age, active = res_val
        x_ct, y_ct = ct 
        x_trk, y_trk = tracking
        xmin, ymin, xmax, ymax = bbox
        
        x_ct_bbox = (xmin + xmax) / 2
        y_ct_bbox = (ymin + ymax) / 2
        
        res_arr[num] = np.array([num, frame_index, class_id, 
                                 x_ct, y_ct, xmin, ymin, xmax, ymax, 
                                 x_ct_bbox, y_ct_bbox, 
                                 tracking_id, score, x_trk, y_trk, age, active])
    parse_end_time = time.time()
    
    person_values = res_arr[res_arr[:, 2]==1]
    ball_values = res_arr[res_arr[:, 2]==2]
    court_values = res_arr[res_arr[:, 2]==3]
    
    ball_data.update(ball_values)
    court_data.update(court_values)
    
    # draw
    court_data.draw(frame)
    ball_data.draw(frame)
            
    infer_fps = round(1 / (infer_end_time - infer_start_time), 1)
    infer_text = 'infer: {} fps'.format(infer_fps)
    
    cv2.putText(frame, infer_text, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (255, 255, 255), thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(frame, infer_text, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_index += 1
    
    datas.append(res_arr)

    
cap.release()
cv2.destroyAllWindows()
# -




