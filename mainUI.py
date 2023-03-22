# Copyright 2016-2022 NXP
# SPDX-License-Identifier: MIT

import os
from shutil import copy,rmtree
import numpy as np
import tarfile
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QThread,QTimer
from PyQt5.QtWidgets import QMainWindow,QApplication,QAction,qApp,QMessageBox
from PyQt5.QtCore import QDateTime,QDate,QTime,Qt,QRect
from PyQt5.QtGui import QIcon,QImage,QPixmap,QPainter, QPen,QColor,QFont,QCursor
import cv2
import lans
from yolo.YOLOv7 import image_object_detect
from yolo.utils import class_names as voc_class_name
from dialg_detect_win import Ui_Dialog,video_Dialog
import config_util
import pascal_voc_writer
import thread

rng = np.random.default_rng(5)
colors = rng.uniform(0, 255, size=(len(voc_class_name), 3))
box_size_rang_str = ['5%~80%','10%~80%','10%~70%','10%~50%']
box_size_rang = [[0.1,0.2],[0.1,0.3],[0.1,0.4],[0.1,0.5]]

def get_images(path):
    list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg' or os.path.splitext(file)[1] == '.jpg':
                list.append(os.path.join(root, file))
                

    return list

class Ui_MainWindow(object):

    def __init__(self,parent = None):
        super(Ui_MainWindow, self).__init__()
        
        '''
        stored in config file
        '''
        self.config = config_util.config()

        if (self.config.load_prev_cfg() == False):
            self.config.lang = 'CNS'
            self.config.class_names = ''
            
            self.config.images_path = 'c:'
            self.config.voc_type = True
            self.config.src_images_idx = 0

            self.config.src_images = []
            self.config.images_box_list = []
            self.config.images_label_list = []
            self.current_boxes = []
            self.current_class_ids = []
            self.current_boxes_en = []
            self.boxes_range = [0,1]
            self.cfg_file = './output/temp.json'
            self.load_flag = False
        else:
            if(len( self.config.src_images ) >  0):
                #
                self.current_boxes = self.config.images_box_list[self.config.src_images_idx]
                self.current_class_ids = self.config.images_label_list[self.config.src_images_idx]
                self.current_boxes_en = self.config.images_box_en_list[self.config.src_images_idx]
                self.boxes_range = [self.config.box_erea_min_filter,self.config.box_erea_max_filter]
                self.cfg_file = self.config.cfg_file
                self.refresh_flag = True
                self.load_flag = True
                #self.progressBar.setValue(int(self.config.src_images_idx/len(self.config.src_images) * 100) +1)

                #self.mainwindow.setWindowTitle("Auto Label  %s"%os.path.basename(self.cfg_file))
                #self.textBrowser.setText("Found %d images"%len(self.config.src_images))
        #load

        '''
        
        '''
        
        self.dict = lans.g_dict[self.config.lang]
        self.video = ''
        self.refresh_flag = False
        #mouse draw box flags
        self.draw_box_x0 = 0
        self.draw_box_x1 = 0
        self.draw_box_y0 = 0
        self.draw_box_y0 = 0
        self.draw_box_mouse_press_flag = 0
        self.draw_box_mouse_move_flag = 0

        self.mouse_select_id = -1 
        self.image_posx = 0
        self.image_posy = 0
        model_path = "yolo/yolov7-tiny_480x640.onnx"
        self.detector = image_object_detect(model_path)

        if os.path.exists('./output') == False:
            os.mkdir('./output')

    def closeEvent(self, event):
        self.config.save_current(self.cfg_file)
        event.accept()    

    def showimage(self,img):
        self.statusLabel.setText(img)
        img = cv2.imread(img)
        try:
            img = cv2.resize(img,(self.image_w, self.image_h))
        
            img_height, img_width, channels = img.shape
            bytesPerLine = channels * self.image_w
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB,img)
            self.image = QImage(img.data,self.image_w, self.image_h,bytesPerLine,QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(self.image))
        except Exception as e:
            print("error: %s"%str(e))
            return False

    def nextButtonFunc(self):
        if(len( self.config.src_images ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有找到jpg图片"])
            return
        self.config.src_images_idx = (self.config.src_images_idx + 1)%len(self.config.src_images)
        
        status = self.showimage(self.config.src_images[self.config.src_images_idx])
        if status == False:
            return
        self.current_boxes = self.config.images_box_list[self.config.src_images_idx]
        self.current_class_ids = self.config.images_label_list[self.config.src_images_idx]
        self.current_boxes_en = self.config.images_box_en_list[self.config.src_images_idx]
        self.progressBar.setValue(int(self.config.src_images_idx/len(self.config.src_images) * 100) +1)
        self.refresh_flag = True
        self.mouse_select_id = -1
        self.update_listview()
        self.savecfg_func()

    def prevButtonFunc(self):
        if(len( self.config.src_images ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有找到jpg图片"])
            return
        if self.config.src_images_idx > 0:
            self.config.src_images_idx = self.config.src_images_idx -1
        else :
            self.config.src_images_idx = len(self.config.src_images) - 1
        
        status = self.showimage(self.config.src_images[self.config.src_images_idx])
        if status == False:
            return
        self.current_boxes = self.config.images_box_list[self.config.src_images_idx]
        self.current_class_ids = self.config.images_label_list[self.config.src_images_idx]
        self.current_boxes_en = self.config.images_box_en_list[self.config.src_images_idx]
        self.refresh_flag = True
        self.progressBar.setValue(int(self.config.src_images_idx/len(self.config.src_images) * 100) +1)
        self.mouse_select_id = -1
        self.update_listview()
        self.savecfg_func()

    def mouseMoveEvent(self,QMouseEvent):
        x = QMouseEvent.x() - self.image_posx
        y = QMouseEvent.y() - self.image_posy
        
        if(len( self.config.src_images ) == 0):
            return
        if x > 0 and x < self.image_w and y > 0 and y < self.image_h:
            self.draw_box_mouse_move_flag = True
            if self.draw_box_mouse_press_flag == True:
                self.draw_box_x1 = QMouseEvent.x() 
                self.draw_box_y1 = QMouseEvent.y()
            
            else:
                self.draw_box_x0 = QMouseEvent.x() 
                self.draw_box_y0 = QMouseEvent.y()

            self.showimage(self.config.src_images[self.config.src_images_idx])
            self.refresh_flag = True
        elif self.draw_box_mouse_move_flag == True:
            self.draw_box_mouse_move_flag = False
            self.showimage(self.config.src_images[self.config.src_images_idx])
            self.refresh_flag = True
            
    def mouse_release_menu_cancel_func(self):
        if self.draw_box_mouse_press_flag == True and self.draw_box_mouse_move_flag == True:
            self.draw_box_mouse_press_flag = False
            self.draw_box_mouse_move_flag = False
            self.refresh_flag = True
            
            self.showimage(self.config.src_images[self.config.src_images_idx])

    def mouse_release_menu_func(self,m):
        if self.draw_box_mouse_press_flag == True and self.draw_box_mouse_move_flag == True:
            class_name = self.centralwidget.sender().objectName()
            if class_name == '':
                return 


            if class_name in self.config.class_names:
                class_id = self.config.class_names.index(class_name)

                boxes = np.array([self.draw_box_x0,self.draw_box_y0,self.draw_box_x1 ,self.draw_box_y1])
                boxes -= np.array([self.image_posx,self.image_posy,self.image_posx,self.image_posy])
                boxes = boxes.astype('float32') / np.array([self.image_w, self.image_h,self.image_w, self.image_h]).astype('float32')
                if self.current_boxes is None:
                    self.current_boxes = []
                    self.current_class_ids = []
                    self.current_boxes_en = []
                
                self.current_boxes.append(boxes.tolist())
                self.current_class_ids.append(class_name)
                self.current_boxes_en.append(True)
                self.config.images_box_list[self.config.src_images_idx] = self.current_boxes 
                self.config.images_label_list[self.config.src_images_idx] = self.current_class_ids 
                self.config.images_box_en_list[self.config.src_images_idx] = self.current_boxes_en
            self.update_listview()
            self.draw_box_mouse_press_flag = False
            self.draw_box_mouse_move_flag = False
            self.refresh_flag = True

    def mouseReleaseEvent(self,QMouseEvent):
        if self.draw_box_mouse_press_flag == True and self.draw_box_mouse_move_flag == True:
            release_menu = QtWidgets.QMenu(self)
            for class_id in self.config.class_names:
                action_name = 'action_' + class_id.strip()
                locals()[action_name] = QAction(class_id, release_menu)
                locals()[action_name].setObjectName(class_id)
                locals()[action_name] .triggered.connect(self.mouse_release_menu_func)
                release_menu.addAction(locals()[action_name] )

            release_menu.addSeparator()
            cancel_action = QAction('Cancel',release_menu)
            cancel_action.triggered.connect(self.mouse_release_menu_cancel_func)
            release_menu.addAction(cancel_action)
            release_menu.popup(QCursor.pos())
            release_menu.show()   
            
            
    def mousePressEvent(self,QMouseEvent):
        x = QMouseEvent.x() - self.image_posx
        y = QMouseEvent.y() - self.image_posy
        

        if (self.draw_box_mouse_press_flag == False and QMouseEvent.buttons() == QtCore.Qt.LeftButton
            and x > 0 and x < self.image_w and y > 0 and y < self.image_h):
            self.draw_box_mouse_press_flag = True
            self.draw_box_x0 = x
            self.draw_box_y0 = QMouseEvent.y()-30


    def paintEvent(self,event):
       
        if (self.refresh_flag == True):    
            self.refresh_flag = False
            idx = 0
            painter = QPainter(self.image)
            
            if self.draw_box_mouse_press_flag == True and self.draw_box_mouse_move_flag == True:
                color = QColor(255,0,0)
                x = self.draw_box_x0 - self.image_posx
                y = self.draw_box_y0 - self.image_posy
                w = abs(self.draw_box_x1-self.draw_box_x0)
                h = abs(self.draw_box_y1-self.draw_box_y0)
                rect = QRect(x,y,w,h)
                painter.setPen(QPen(color, 4, Qt.SolidLine))
                painter.drawRect(rect)      
                self.image_label.setPixmap(QPixmap.fromImage(self.image))
            elif self.draw_box_mouse_move_flag == True and self.draw_box_mouse_press_flag == False:
                color = QColor(255,0,0)
                x = self.draw_box_x0 - self.image_posx
                y = self.draw_box_y0 - self.image_posy-30
                painter.setPen(QPen(color, 4, Qt.SolidLine))
                painter.drawLine(x,0,x,self.image_h)
                painter.drawLine(0,y,self.image_w,y)
                self.image_label.setPixmap(QPixmap.fromImage(self.image))

            if(self.current_boxes is  not None and self.current_class_ids is not None):         


                for box, class_id ,enable in zip(self.current_boxes, self.current_class_ids,self.current_boxes_en):
                    if(enable == False):
                        continue
                    x1, y1, x2, y2 = (box*np.array([self.image_w, self.image_h, self.image_w, self.image_h])).astype(int)
                    idx = self.config.class_names.index(class_id)
                    color = colors[idx].astype(int)

                    color = QColor(color[0],color[1],color[2])
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    rect = QRect(x, y, w,h)
                    idx = self.current_boxes.index(box)
                    if(idx == self.mouse_select_id):
                        painter.setBrush(QColor(255, 0, 0, 128))
                    else:
                        painter.setBrush(QColor(0, 0, 0, 0))
                        
                    painter.setFont(QFont('SimDun',10))
                    painter.setPen(QPen(color, 4, Qt.SolidLine))
                    painter.drawRect(rect)
                    painter.drawText(rect,Qt.AlignCenter,class_id)
                    idx = idx +1    

            self.image_label.setPixmap(QPixmap.fromImage(self.image))

    def listview_click(self):
        idxs = self.boxes_ListView.selectedIndexes()
        for idx in idxs:
            self.mouse_select_id = (idx.row())
            self.showimage(self.config.src_images[self.config.src_images_idx])
            self.refresh_flag = True
        

    def box_list_checkbox(self,idx):
        count = self.boxes_ListView.count()
        for i in range(count):
            item = self.boxes_ListView.item(i)
            widget = self.boxes_ListView.itemWidget(item)
            
            self.current_boxes_en[i] = widget.isChecked()
        self.listview_click()    
        
    def update_listview(self):
        self.boxes_ListView.clear()
        if(self.current_boxes is None or self.current_class_ids is None):
            return
        
        for i in range(len(self.current_class_ids)):
            class_id = self.current_class_ids[i]
            box = QtWidgets.QCheckBox(class_id + '_%d'%i)
            try:
                if self.current_boxes_en[i] == True:
                    box.setChecked(True)
            except:
                print()
            item = QtWidgets.QListWidgetItem()
            self.boxes_ListView.addItem(item)
            self.boxes_ListView.setItemWidget(item,box)
            box.stateChanged.connect(self.box_list_checkbox)
    
        
        self.boxes_ListView.itemClicked.connect(self.listview_click)


    def labelAllThreadFunc(self,config):
        self.refresh_flag = True
        self.config = config
        self.showimage(self.config.src_images[self.config.src_images_idx])

        self.current_boxes = self.config.images_box_list[self.config.src_images_idx] 
        self.current_class_ids = self.config.images_label_list[self.config.src_images_idx]
        self.current_boxes_en = self.config.images_box_en_list[self.config.src_images_idx]

        self.update_listview()
        self.progressBar.setValue(int(self.config.src_images_idx*100/len(self.config.src_images)))
        if(self.config.src_images_idx +1 == len(self.config.src_images)):
            self.prevButton.setDisabled(False)
            self.nextButton.setDisabled(False)
            self.labelAllButton.setDisabled(False)
            self.labelButton.setDisabled(False)
            self.setMouseTracking(True)
            self.textBrowser.setText("Labled %d images"%(len(self.config.src_images)))
            QMessageBox.warning(self,"Information" , "Auto label success")

    def labelAllFunc(self):
        if(len( self.config.src_images ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有找到jpg图片"])
            return
        
        if(len( self.config.class_names ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有设定标签，新建或打开配置文件"])
            return
          
        self.config.src_images_idx = 0
        self.prevButton.setDisabled(True)
        self.nextButton.setDisabled(True)
        self.labelAllButton.setDisabled(True)
        self.labelButton.setDisabled(True)
        self.setMouseTracking(False)
        self.labelthread = thread.label_thread(self.config,self.detector,self.boxes_range)
        self.labelthread.update_signal.connect(self.labelAllThreadFunc)
        self.labelthread.start()

    def labelButtonFunc(self):
        
        if(len( self.config.src_images ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有找到jpg图片"])
            return
        
        if(len( self.config.class_names ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有设定标签，新建或打开配置文件"])
            return
        
        img = cv2.imread(self.config.src_images[self.config.src_images_idx])
        boxs, scores,names = self.detector.detect(img)
        valid_boxes = []
        valid_labels=[]
        en_boxes = []

        small_box_count = 0
        for i in range(len(boxs)):
            if(names[i] in self.config.class_names):
                area = abs(boxs[i][0] - boxs[i][2])*abs(boxs[i][1]-boxs[i][3])
                if area >= self.boxes_range[0] and area <= self.boxes_range[1]:
                    valid_boxes.append(boxs[i].tolist())
                    valid_labels.append(names[i])
                    en_boxes.append(True)
                else:
                    small_box_count += 1
                    #print("box area size out of range:%f%%\r\n"%(area*100))
        log_str = 'Found %d boxes.\r\n Aviribel :%d, %d out of size.'%(len(en_boxes)+small_box_count, len(en_boxes),small_box_count)
        self.textBrowser.setText(log_str)
        self.refresh_flag = True
        self.showimage(self.config.src_images[self.config.src_images_idx])
        self.config.images_box_list[self.config.src_images_idx] = self.current_boxes = valid_boxes
        self.config.images_label_list[self.config.src_images_idx] = self.current_class_ids = valid_labels
        self.config.images_box_en_list[self.config.src_images_idx] = self.current_boxes_en = en_boxes

        self.update_listview()


    def opendir_func(self,filepath):
        self.config.images_path  = QtWidgets.QFileDialog.getExistingDirectory(None,self.dict["选择图片文件夹"],self.config.images_path)
        self.config.src_images = get_images(self.config.images_path)
        self.config.images_box_list = [None]*len(self.config.src_images)
        self.config.images_label_list = [None]*len(self.config.src_images)
        self.config.images_box_en_list = [None]*len(self.config.src_images)
        if(len( self.config.src_images ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有找到jpg图片"])
            return
        self.config.src_images_idx = 0
        self.current_boxes = self.config.images_box_list[self.config.src_images_idx]
        self.current_class_ids = self.config.images_label_list[self.config.src_images_idx]
        self.current_boxes_en = self.config.images_box_en_list[self.config.src_images_idx]      
        self.showimage(self.config.src_images[0])
        self.textBrowser.setText("Found %d images"%len(self.config.src_images))
        

    def openvideo_func(self):
        video = video_Dialog(self)
        video.setupUi()
        video.show() 
        video.dialogSignel.connect(self.getvideo_from_dialog)
        
    def getvideo_from_dialog(self,flag,path):
        if(flag == 1):
            self.config.images_path = path
            self.config.src_images = get_images(self.config.images_path)
            self.config.images_box_list = [None]*len(self.config.src_images)
            self.config.images_label_list = [None]*len(self.config.src_images)
            self.config.images_box_en_list = [None]*len(self.config.src_images)
            if(len( self.config.src_images ) == 0):
                QMessageBox.critical(self,self.dict["错误"],self.dict["没有找到jpg图片"])
                return
            self.config.src_images_idx = 0
            self.current_boxes = self.config.images_box_list[self.config.src_images_idx]
            self.current_class_ids = self.config.images_label_list[self.config.src_images_idx]
            self.current_boxes_en = self.config.images_box_en_list[self.config.src_images_idx]      
            self.showimage(self.config.src_images[0])

    def opencfg_func(self):
        cfg,_ = QtWidgets.QFileDialog.getOpenFileName(self, self.dict["打开工程"], '.', '(*.json)')
        self.cfg_file = cfg
        error_msg = self.config.load(cfg)
        if(error_msg is not None):
            self.textBrowser.setText(error_msg)

        if(len( self.config.src_images ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有找到jpg图片"])
            return

        self.showimage(self.config.src_images[self.config.src_images_idx])
        self.current_boxes = self.config.images_box_list[self.config.src_images_idx]
        self.current_class_ids = self.config.images_label_list[self.config.src_images_idx]
        self.current_boxes_en = self.config.images_box_en_list[self.config.src_images_idx]
        self.boxes_range = [self.config.box_erea_min_filter,self.config.box_erea_max_filter]

        self.refresh_flag = True
        self.load_flag = True
        self.progressBar.setValue(int(self.config.src_images_idx/len(self.config.src_images) * 100) +1)
        self.update_listview()
        self.mainwindow.setWindowTitle("Auto Label  %s"%os.path.basename(self.cfg_file))
        self.textBrowser.setText("Found %d images"%len(self.config.src_images))

    def getcfg_from_dialog(self,single,list):
        
        if(self.config.class_names != list[0]):
            self.current_class_ids = []
            self.current_boxes = []
            self.current_boxes_en = []
        self.config.class_names = list[0]
        self.config.box_erea_min_filter = list[1]/100
        self.config.box_erea_max_filter = list[2]/100
        self.boxes_range = [list[1]/100,list[2]/100]
        self.config.voc_type = list[3]
        print(self.boxes_range)

    def savecfg_func(self):
        self.config.save(self.cfg_file)

    def editcfg_func(self):
        self.diag = Ui_Dialog(self)
        self.diag.setUI(voc_class_name,self.config.box_erea_min_filter,self.config.box_erea_max_filter,self.config.voc_type)
        if (self.config.class_names != ''):
            ret = '; '.join(self.config.class_names)
            self.diag.set_selected_object(ret)
        
        self.diag.show()
        
        self.diag.dialogSignel.connect(self.getcfg_from_dialog)
        self.mainwindow.setWindowTitle("Auto Label  %s"%os.path.basename(self.cfg_file))
    def newcfg_func(self):
        cfg,_ = QtWidgets.QFileDialog.getSaveFileName(self, self.dict["新建工程"], '.', '(*.json)')
        self.cfg_file = cfg
        
        self.diag = Ui_Dialog(self)
        self.diag.setUI(voc_class_name,self.config.box_erea_min_filter,self.config.box_erea_max_filter,self.config.voc_type)
        if (self.config.class_names != ''):
            ret = '; '.join(self.config.class_names)
            self.diag.set_selected_object(ret)
        if self.load_flag == True:
            self.image_label.clear()
            self.statusLabel.clear()
            self.config = config_util.config()
            self.config.src_images = []
        self.diag.show()
        
        self.diag.dialogSignel.connect(self.getcfg_from_dialog)
        self.mainwindow.setWindowTitle("Auto Label  %s"%os.path.basename(self.cfg_file))
        
        print('save cfg')

    def switch_cns(self):
        if self.config.lang != 'CNS':
            self.config.lang = 'CNS'
            self.savecfg_func()
            self.dict = lans.g_dict[self.config.lang]
            self.retranslateUi(self.mainwindow)
            self.lans_en.setChecked(False)
            self.lans_cns.setChecked(True)
            
            #self.setupUi(self.mainwindow)
            #self.show()
            #Todo 
    def switch_en(self):
        if self.config.lang != 'EN':
            self.config.lang = 'EN'
            self.savecfg_func()
            self.dict = lans.g_dict[self.config.lang]
            self.retranslateUi(self.mainwindow)
            self.lans_en.setChecked(True)
            self.lans_cns.setChecked(False)
            #self.setupUi(self.mainwindow)
            #self.show()
            #Todo 
            #      
            # 
            
    def tar_progress_func(self,value,cmd,log):
        if cmd == 'inprogress':
            self.progressBar.setValue(value)
        elif cmd == 'end':
            self.progressBar.setValue(100)
            self.prevButton.setDisabled(False)
            self.nextButton.setDisabled(False)
            self.labelAllButton.setDisabled(False)
            self.labelButton.setDisabled(False)
            self.setMouseTracking(True)
            self.textBrowser.setText(log)
            self.savecfg_func()
            QMessageBox.warning(self,"Information" , "Export success")

    def export_func(self):
        if(len( self.config.src_images ) == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有找到jpg图片"])
            return
        labels_count = 0
        for labels in self.config.images_box_list:
            if labels is not None:
                labels_count = labels_count +1

        if(labels_count == 0):
            QMessageBox.critical(self,self.dict["错误"],self.dict["没有标注图片"])
            return

        self.voc_file,_ = QtWidgets.QFileDialog.getSaveFileName(self, self.dict['保存voc文件'], '.', '(*.tar)')

        self.prevButton.setDisabled(True)
        self.nextButton.setDisabled(True)
        self.labelAllButton.setDisabled(True)
        self.labelButton.setDisabled(True)
        self.setMouseTracking(False)
        self.tarthread = thread.tar_thread(self.config,self.voc_file)
        self.tarthread.update_signal.connect(self.tar_progress_func)
        self.tarthread.start()
        


    def setupUi(self, MainWindow):

        MainWindow.setObjectName("Auto Lable")
        MainWindow.resize(1200, 800)

        self.opendir = QAction(self.dict['打开图片文件夹'], self)
        self.opendir.triggered.connect(self.opendir_func)
        self.openvideo = QAction(self.dict['打开视频'], self)
        self.openvideo.triggered.connect(self.openvideo_func)

        self.opencfg = QAction(self.dict['打开工程'], self)
        self.opencfg.triggered.connect(self.opencfg_func)
        
        self.newcfg = QAction(self.dict['新建工程'], self)
        self.newcfg.triggered.connect(self.newcfg_func)

        self.editcfg = QAction(self.dict['编辑工程'], self)
        self.editcfg.triggered.connect(self.editcfg_func)

        self.savecfg = QAction(self.dict['保存工程'], self)
        self.savecfg.triggered.connect(self.savecfg_func)

        self.lans_cns = QAction(self.dict['中文'], self)
        self.lans_cns.triggered.connect(self.switch_cns)
        self.lans_cns.setCheckable(True)
        self.lans_en = QAction(self.dict['英文'], self)
        self.lans_en.triggered.connect(self.switch_en)
        self.lans_en.setCheckable(True)
        if(self.config.lang == 'EN'):
            self.lans_en.setChecked(True)
        else:
            self.lans_cns.setChecked(True)

        self.export = QAction(self.dict['保存voc文件'], self)
        self.export.triggered.connect(self.export_func)
        self.mainwindow = MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.statusLabel = QtWidgets.QLabel(self.centralwidget)
        self.statusLabel.setGeometry(QtCore.QRect(0, 0, 800, 40))
        self.statusLabel.setStyleSheet("font-size: 12px; font-weight: bold; height: 40px; margin-left: 5px;vertical-align:middle")
        self.statusLabel.setText('Blank Project')

        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(0, 40, 800, 640))
        self.image_label.setObjectName("image_label")
        self.image_label.setMouseTracking(True)
        self.centralwidget.setMouseTracking(True)
        self.setMouseTracking(True)
        #self.setCursor(Qt.CrossCursor)
        self.image_posx=0
        self.image_posy=40
        self.image_w=800
        self.image_h=640

        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(0, 680, 800, 31))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setRange(0,100)
        self.progressBar.setValue(0)

        self.prevButton = QtWidgets.QPushButton(self.centralwidget)
        self.prevButton.setGeometry(QtCore.QRect(840, 190, 120, 28))
        self.prevButton.setObjectName("prevButton")
        self.nextButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextButton.setGeometry(QtCore.QRect(840, 240, 120, 28))
        self.nextButton.setObjectName("nextButton")

        self.labelButton = QtWidgets.QPushButton(self.centralwidget)
        self.labelButton.setGeometry(QtCore.QRect(1020 , 190, 120, 28))
        self.labelButton.setObjectName("labelButton")

        self.labelAllButton = QtWidgets.QPushButton(self.centralwidget)
        self.labelAllButton.setGeometry(QtCore.QRect(1020, 240, 120, 28))
        self.labelAllButton.setObjectName("labellAllButton")

        
        self.boxes_ListView = QtWidgets.QListWidget(self.centralwidget)
        self.boxes_ListView.setGeometry(QtCore.QRect(840, 300, 300, 360))

        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(840,40,300,120))
        

        self.nextButton.clicked.connect(self.nextButtonFunc)
        self.prevButton.clicked.connect(self.prevButtonFunc)
        self.labelButton.clicked.connect(self.labelButtonFunc)
        self.labelAllButton.clicked.connect(self.labelAllFunc)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1059, 26))
        self.menubar.setObjectName("menubar")

        self.fileMenu = self.menubar.addMenu(self.dict['文件'])
        self.importMenu = self.menubar.addMenu(self.dict['导入'])
        self.exportMenu = self.menubar.addMenu(self.dict['导出数据集'])
        self.langMenu = self.menubar.addMenu(self.dict['语言'])

        self.fileMenu.addAction(self.newcfg)
        self.fileMenu.addAction(self.opencfg)
        self.fileMenu.addAction(self.editcfg)
        self.fileMenu.addAction(self.savecfg)

        self.importMenu.addAction(self.opendir)
        self.importMenu.addAction(self.openvideo)
        self.exportMenu.addAction(self.export)
        self.langMenu.addAction(self.lans_cns)
        self.langMenu.addAction(self.lans_en)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        if self.load_flag == True:
            self.showimage(self.config.src_images[self.config.src_images_idx])

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Auto Label"))
        
        self.prevButton.setText(_translate("MainWindow", "Prev"))
        self.nextButton.setText(_translate("MainWindow", "Next"))
        self.labelButton.setText(_translate("MainWindow", "Auto Label"))
        self.labelAllButton.setText(_translate("MainWindow", "Auto Label All"))

        self.fileMenu.setTitle(_translate("MainWindow", self.dict['文件']))
        self.importMenu.setTitle(_translate("MainWindow", self.dict['导入']))
        self.exportMenu.setTitle(_translate("MainWindow", self.dict['导出数据集']))


        self.opendir.setText(_translate("MainWindow", self.dict['打开图片文件夹']))
        self.openvideo.setText(_translate("MainWindow", self.dict['打开视频']))
        self.opencfg.setText(_translate("MainWindow", self.dict['打开工程']))
        self.newcfg.setText(_translate("MainWindow", self.dict['新建工程']))
        self.editcfg.setText(_translate("MainWindow", self.dict['编辑工程']))
        self.savecfg.setText(_translate("MainWindow", self.dict['保存工程']))
        self.export.setText(_translate("MainWindow", self.dict['保存voc文件']))

        

import sys
class MainCode(QMainWindow,Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
      
if __name__=='__main__':
    app=QApplication(sys.argv)
    md=MainCode()
    md.show()
    sys.exit(app.exec_())
