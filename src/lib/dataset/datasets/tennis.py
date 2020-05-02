from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .coco import COCO
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import copy

from ..generic_dataset import GenericDataset

class Tennis(COCO):
  default_resolution = [512, 512]
  num_categories = 3
  class_name = [
    'person', 'ball', 'court_edge']
  _valid_ids = [
      1, 2, 3
      ]
  def __init__(self, opt, split, data_name=None):
    # load annotations
    if data_name is None:
      data_dir = os.path.join(opt.data_dir, 'coco')
    else:
      data_dir = os.path.join(opt.data_dir, 'coco', data_name)

    img_dir = os.path.join(data_dir, '{}2017'.format(split))
    if opt.trainval:
      split = 'test'
      ann_path = os.path.join(
          data_dir, 'annotations', 
          'image_info_test-dev2017.json')
    else:
        ann_path = os.path.join(
          data_dir, 'annotations', 
          'instances_{}2017.json').format(split)

    self.images = None
    # load image list and coco
    super(COCO, self).__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      if type(all_bboxes[image_id]) != type({}):
        # newest format
        for j in range(len(all_bboxes[image_id])):
          item = all_bboxes[image_id][j]
          cat_id = item['class'] - 1
          category_id = self._valid_ids[cat_id]
          bbox = item['bbox']
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(item['score']))
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results_coco.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results_coco.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()