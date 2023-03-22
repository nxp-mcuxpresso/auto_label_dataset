# Copyright 2016-2022 NXP
# SPDX-License-Identifier: MIT
import cv2
from YOLOv7 import YOLOv7

# Initialize YOLOv7 object detector
model_path = "yolov7-tiny_480x640.onnx"
class image_object_detect():

    def __init__(self):
        self.yolov7_detector = YOLOv7(model_path, conf_thres=0.3, iou_thres=0.5)

    def detect(self,cv_img):
        boxes, scores, class_ids = self.yolov7_detector(cv_img)
        out_img = self.yolov7_detector.draw_detections(cv_img)
        return out_img

