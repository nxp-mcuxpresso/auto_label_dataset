# Copyright 2016-2022 NXP
# SPDX-License-Identifier: MIT
import json
import os
from queue import Empty
import numpy as np

class config:
    def __init__(self):
        self.lang = 'CNS'
        self.class_names = ''
        self.box_erea_min_filter = 0
        self.box_erea_max_filter = 1
        self.images_path = 'c:'
        self.voc_type = True
        self.src_images_idx = 0

        self.src_images = []
        self.images_box_list = []
        self.images_label_list = []
        self.images_box_en_list = []
        self.project_name = 'temp'
        self.cache_file = './output/cache.json'
        self.cfg_file = './output/temp.json'
    def load_prev_cfg(self):
        try:
            f = open(self.cache_file,'r')
            dict_str = f.read()
            f.close()
            dict = json.loads(dict_str)
            self.cfg_file = dict['config_file']
            error_msg = self.load(self.cfg_file)
            if(error_msg is not None):
                return False
            return True
        except Exception as e:
            print("Error {0}".format(str(e)))
            return False


    def save_current(self,cfg_file):
        try:
            f = open(self.cache_file,'w')
            dict = {'config_file':cfg_file}
            dict_str = json.dumps(dict)
            f.write(dict_str)
            f.close()
        except Exception as e:
            print("Error {0}".format(str(e)))

    def load(self,file):
        
        try:
            f = open(file,'r')
            dict_str = f.read()
            f.close()
            dict = json.loads(dict_str)

            self.src_images = []
            self.images_box_list = []
            self.images_label_list = []
            self.images_box_en_list = []
            self.lang = dict['lang']
            self.images_path = dict['path']
            self.class_names = dict['labels']
            self.box_erea_min_filter = dict['box_erea_min_filter']
            self.box_erea_max_filter = dict['box_erea_max_filter']
            self.voc_type = dict['default_types']
            self.src_images_idx = dict['current_idx']
            self.project_name = file[file.rfind('/')+1:file.rfind('.')]
            images_config = './output/%s_images.json'%self.project_name
            try:
                f = open(images_config,'r')
                dict_str = f.read()
                dicts = json.loads(dict_str)
                f.close()
            except Exception as e:
                print('load images config error:%s'%str(e))
                dicts = []
            if dict is not Empty:
                for dict in dicts:
                    self.src_images.append(dict['image'])
                    labels = dict['objects']
                    boxes = dict['boxes']
                    enables = dict['enables']
                    self.images_box_list.append(boxes)
                    self.images_label_list.append(labels)
                    self.images_box_en_list.append(enables)
        except Exception as e:
            self.__init__()
            return ("Error {0}".format(str(e)))
        
    
    def save(self,file):
        if len(self.src_images) != len(self.images_box_list) or len(self.src_images) != len(self.images_label_list) :
            return -1
        dict = {'lang':self.lang,'path':self.images_path,'labels':self.class_names,'box_erea_min_filter':self.box_erea_min_filter
                ,'box_erea_max_filter':self.box_erea_max_filter,'default_types':self.voc_type,'current_idx':self.src_images_idx}
        try:
            dict_str = json.dumps(dict)
            f = open(file,'w')
            f.write(dict_str)
            f.close()
        except:
            print('save config file error')
            return

        self.project_name = file[file.rfind('/')+1:file.rfind('.')]
        images_config = './output/%s_images.json'%self.project_name
        dict_list = []
        for i in range(len(self.src_images)):
            if(self.images_box_list[i] is not None):
                boxes = (self.images_box_list[i])
                labels = (self.images_label_list[i])
                enables = (self.images_box_en_list[i])
            else:
                boxes = None
                labels = None
                enables = None
            dict = {'image':self.src_images[i],'objects':labels,'boxes':boxes,'enables':enables}
            dict_list.append(dict)

        try:
            dict_str = json.dumps(dict_list)
            f = open(images_config,'w')
            f.write(dict_str)
            f.close()
        except Exception as e:
            print('write images config error:%s'%str(e))         
            #return -1

