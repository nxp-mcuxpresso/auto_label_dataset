# Copyright 2016-2022 NXP
# SPDX-License-Identifier: MIT
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.Qt import QThread,QTimer
from PyQt5.QtWidgets import QMainWindow,QApplication,QAction,qApp,QMessageBox
import config_util
import os
import cv2
from shutil import copy,rmtree
import pascal_voc_writer
import tarfile
import numpy as np

class tar_thread(QThread):
    update_signal = pyqtSignal(int,str,str)
    config = config_util.config()

    def __init__(self, config,voc_file,parent=None):
        super(tar_thread, self).__init__(parent)
        self.config = config
        self.voc_file = voc_file

    def run(self):
        voc_name = os.path.basename(self.voc_file)
        voc_name = voc_name[0:voc_name.rfind('.')]
        try:
            os.mkdir('./output/%s'%voc_name)
            os.mkdir('./output/%s/Annotations'%voc_name)
            os.mkdir('./output/%s/JPEGImages'%voc_name)
            os.mkdir('./output/%s/ImageSets'%voc_name)
            os.mkdir('./output/%s/Json_Annos'%voc_name)
            os.mkdir('./output/%s/tools'%voc_name)
        except Exception as e:
            print(e)

        images_count = 0
        empty_count = 0
        progress = 0
        for i in range(len(self.config.src_images)):
            current = images_count*90 / len(self.config.src_images)
            if int(current) > progress:
                progress = int(current)
                self.update_signal.emit(progress,'inprogress','')
            
            if self.config.images_box_list[i] is not None and len(self.config.images_box_list[i]) > 0:
                img = self.config.src_images[i]
                img_name = os.path.basename(img)
                #if(os.path.exists('./output/%s/JPEGImages/%s'%(voc_name,img_name))) == True:
                #    print('./output/%s/JPEGImages/%s'%(voc_name,img_name))
                im = cv2.imread(img)
                img_height, img_width, channels = im.shape
                abspath = os.path.abspath(img)
                st = copy(abspath,'./output/%s/JPEGImages/%s'%(voc_name,img_name))
                
                w = pascal_voc_writer.Writer(img,img_width,img_height)
                for box, class_id ,enable in zip(self.config.images_box_list[i], self.config.images_label_list[i], self.config.images_box_en_list[i]):
                    if(enable == True):
                        box = (box * np.array([img_width,img_height,img_width,img_height])).astype(int)
                        xmin,ymin,xmax,ymax = box
                        if xmin <= 0:xmin=10
                        if ymin <=0:ymin=10
                        if xmax >= img_width:xmax = img_width-10
                        if ymax >= img_height:ymax=img_height-10
                        w.addObject(class_id,xmin,ymin,xmax,ymax)
                xml_name = img_name[0:img_name.rfind('.')] + '.xml'
                w.save('./output/%s/Annotations/%s'%(voc_name ,xml_name))
                images_count = images_count +1
            else:
                empty_count = empty_count +1
                print(self.config.src_images[i])

        try:
            tar = tarfile.open(self.voc_file,'w')
            for root, dirs, files in os.walk('./output/%s'%voc_name):
                for single_file in files:
                    # if single_file != tarfilename:
                    filepath = os.path.join(root, single_file)
                    dir_name = os.path.basename(os.path.dirname(filepath))
                    tar.add(filepath,voc_name + '/' + dir_name + '/' + single_file)
            tar.close()
            
            log = self.voc_file+" exported, images count %d !"%images_count
            rmtree('./output/%s'%voc_name)
        except Exception as e:
            
            log = self.voc_file+"Error: %s!"%str(e)

        self.update_signal.emit(100,'end',log)


class label_thread(QThread):
    update_signal = pyqtSignal(object)
    config = config_util.config()

    def __init__(self, config,detector,boxes_range,parent=None):
        super(label_thread, self).__init__(parent)
        self.config = config
        self.detector =detector
        self.boxes_range = boxes_range

    def run(self):
        for i in range(len(self.config.src_images)):
            self.config.src_images_idx = i
            img = cv2.imread(self.config.src_images[self.config.src_images_idx])
            
            if img is None:
                print("%s error "%self.config.src_images[self.config.src_images_idx])
                continue
            boxs, scores,names = self.detector.detect(img)
            valid_boxes = []
            valid_labels=[]
            en_boxes = []
            for i in range(len(boxs)):
                if(names[i] in self.config.class_names):
                    area = abs(boxs[i][0] - boxs[i][2])*abs(boxs[i][1]-boxs[i][3])
                    if(boxs[i][0] >= boxs[i][2]) or boxs[i][1] >= boxs[i][3]:
                        print("error box at %d"%i)
                    if area >= self.boxes_range[0] and area <= self.boxes_range[1]:
                        valid_boxes.append(boxs[i].tolist())
                        valid_labels.append(names[i])
                        en_boxes.append(True)
            self.config.images_box_list[self.config.src_images_idx] = valid_boxes
            self.config.images_label_list[self.config.src_images_idx] = valid_labels
            self.config.images_box_en_list[self.config.src_images_idx] = en_boxes
            self.update_signal.emit(self.config)