# Copyright 2016-2022 NXP
# SPDX-License-Identifier: MIT
from queue import Empty
from sre_constants import SUCCESS
import string
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QComboBox, QLineEdit, QListWidgetItem, QListWidget, QCheckBox, \
  QApplication, QVBoxLayout, QWidget,QDialog,QRadioButton,QMessageBox
from PyQt5.QtGui import QIcon,QImage,QPixmap,QPainter, QPen,QColor,QFont,QCursor
from PyQt5.QtCore import pyqtSignal
from PyQt5.Qt import QThread,QTimer
import re
import cv2
import os

import numpy as np

def get_images(path):
    list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg' or os.path.splitext(file)[1] == '.jpg':
                list.append(os.path.join(root, file))
                

    return list

class ComboCheckBox(QComboBox):
    def __init__(self, items: list):

        super(ComboCheckBox, self).__init__()
        self.items = ["Select All"] + items # items list
        self.box_list = [] # selected items
        self.text = QLineEdit() # use to selected items
        self.state = 0 # use to record state
        self.text_string = ''
        

        q = QListWidget()
        for i in range(len(self.items)):
            self.box_list.append(QCheckBox())
            self.box_list[i].setText(self.items[i])
            item = QListWidgetItem(q)
            q.setItemWidget(item, self.box_list[i])
            if i == 0:
                self.box_list[i].stateChanged.connect(self.all_selected)
            else:
                self.box_list[i].stateChanged.connect(self.show_selected)
        q.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
        self.setStyleSheet("width: 20px; height: 30px; font-size: 16px; font-weight: bold;vertical-align:middle")
        self.text.setReadOnly(True)
        self.setLineEdit(self.text)
        self.setModel(q.model())
        self.setView(q)
        
    def set_default_selected(self,def_string):
        for i in range(len(self.items)):
            item = self.items[i]
            if item in def_string:
                self.box_list[i].setChecked(True)
        self.show_selected()

    def all_selected(self):
        """
        decide whether to check all
        :return:
        """
        # change state
        if self.state == 0:
            self.state = 1
        for i in range(1, len(self.items)):
            self.box_list[i].setChecked(True)
        else:
            self.state = 0
        for i in range(1, len(self.items)):
            self.box_list[i].setChecked(False)
        self.show_selected()
    def get_selected(self) -> list:

        ret = []
        for i in range(1, len(self.items)):
            if self.box_list[i].isChecked():
                ret.append(self.box_list[i].text())
        return ret
    
    def get_selected_idx(self) -> list:
    
        ret = []
        for i in range(1, len(self.items)):
            if self.box_list[i].isChecked():
                ret.append(i)
        return ret

    def show_selected(self):
        """
        show selected items
        :return:
        """
        self.text.clear()
        ret = '; '.join(self.get_selected())
        self.text.setText(ret)
        self.text_string = ret

class Ui_Dialog(QDialog):
    
    dialogSignel=pyqtSignal(int,list)

    def __init__(self,parent=None):
        super(Ui_Dialog, self).__init__(parent)
        self.setWindowTitle('Setting')
        self.resize(400, 200)
        self.voc_type = True
        self.objclass_str = ""
        self.boxrange_str = ""
        self.layout = QtWidgets.QFormLayout()
        self.setLayout(self.layout)

    def detroy_layout(self):
        if self.layout is not None:
            
            while self.layout.count():
                item = self.layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                
    def voc_type_radio_button(self):
        self.voc_type = True
        self.user_radio.setChecked(False)
        self.detroy_layout()

        self.setUI(self.objclass_str,self.box_erea_min,self.box_erea_max,self.voc_type)
        

    def user_type_radio_button(self):
        self.voc_type = False
        self.voc_radio.setChecked(False)
        self.detroy_layout()

        self.setUI(self.objclass_str,self.box_erea_min,self.box_erea_max,self.voc_type)

    def set_selected_object(self ,str):
        if(self.voc_type == True):
            self.object_combo.set_default_selected(str)
        else:
            self.input_types.setText(str)

    def setUI(self,objclass_str,box_erea_min,box_erea_max,voc_type=True):    
        
        self.voc_type = voc_type
        self.objclass_str = objclass_str
        self.box_erea_min = box_erea_min
        self.box_erea_max = box_erea_max

        option_label = QtWidgets.QLabel()
        option_label.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
        option_label.setText("Select Dataset Types")
        self.layout.addRow(option_label,None)

        self.voc_radio = QtWidgets.QRadioButton()
        self.voc_radio.setText("Select Objects from COCO")
        self.voc_radio.clicked.connect(self.voc_type_radio_button)

        self.user_radio  = QtWidgets.QRadioButton()
        self.user_radio.setText("User Define Objects")
        
        self.user_radio.clicked.connect(self.user_type_radio_button)
        self.layout.addRow(self.voc_radio,self.user_radio)
        
        if(self.voc_type):
            self.voc_radio.setChecked(True)
            self.user_radio.setChecked(False)
            self.object_combo = ComboCheckBox(objclass_str)
            dialog_label = QtWidgets.QLabel()
            dialog_label.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
            dialog_label.setText("Select types")
            self.layout.addRow(dialog_label,self.object_combo)

        else:
            self.voc_radio.setChecked(False)
            self.user_radio.setChecked(True)
            dialog_label = QtWidgets.QLabel()
            dialog_label.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
            dialog_label.setText("Input types")
            self.input_types = QtWidgets.QLineEdit()
            self.input_types.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
            self.layout.addRow(dialog_label,self.input_types)

        box_label=  QtWidgets.QLabel()
        box_label.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
        box_label.setText("Set Min Box Area Filter")
        self.min_box_erea = QtWidgets.QLineEdit()
        self.min_box_erea.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
        self.layout.addRow(box_label,self.min_box_erea)
        if(box_erea_min <= 1 and box_erea_min >= 0):
            self.min_box_erea.setText('%d%%'%int(box_erea_min*100))
        elif box_erea_min > 0 and box_erea_min <= 100 :
            self.min_box_erea.setText('%d%%'%int(box_erea_min))

        box_label=  QtWidgets.QLabel()
        box_label.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
        box_label.setText("Set Max Box Area Filter")
        self.max_box_erea = QtWidgets.QLineEdit()
        self.max_box_erea.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
        self.layout.addRow(box_label,self.max_box_erea)
        if(box_erea_max <= 1 and box_erea_max >= 0):
            self.max_box_erea.setText('%d%%'%int(box_erea_max*100))
        elif box_erea_max > 0 and box_erea_max <= 100 :
            self.min_box_erea.setText('%d%%'%int(box_erea_max))

        ok_pushbutton = QtWidgets.QPushButton()
        ok_pushbutton.setStyleSheet("font-size: 12px; font-weight: bold; height: 26px; margin-left: 5px;vertical-align:middle")
        ok_pushbutton.setText("Ok")
        ok_pushbutton.clicked.connect(self.okButtonFunc)

        self.layout.addRow(ok_pushbutton)
        

    def okButtonFunc(self):
        list = []
        if self.voc_type == True:
            if len(self.object_combo.get_selected()) == 0:
                QMessageBox.critical(self,"Error","Input types shouldn't be empty !")
                return
            list.append(self.object_combo.get_selected())
        else:
            types = self.input_types.text()
            types = re.split(r'[;, ]',types)
            
            if len(types[0]) == 0:
                QMessageBox.critical(self,"Error","Input types shouldn't be empty !")
                return
            list.append(types)
        box_min_str = self.min_box_erea.text()
        self.box_erea_min = int(box_min_str[0:box_min_str.rfind('%')])
        box_max_str = self.max_box_erea.text()
        self.box_erea_max = int(box_max_str[0:box_max_str.rfind('%')])
        list.append(self.box_erea_min)
        list.append(self.box_erea_max)
        list.append(self.voc_type )
        
        self.dialogSignel.emit(0,list)
        self.close()
        
class video_Dialog(QDialog):
    
    dialogSignel=pyqtSignal(int,str)
    def __init__(self,parent=None):
        super(video_Dialog, self).__init__(parent)
        self.setWindowTitle('Decoding Video')
        self.resize(800, 480)
        self.threshold = 0.75
        self.video = ''
        self.decoded_images = []
        self.decoded_images_idx = 0
        self.dir_name = ''
        self.saved_count = 0

        self.diag = QtWidgets.QWidget(self)
        self.formLayoutWidget = QtWidgets.QWidget(self.diag)
        self.layoutwidget = QtWidgets.QWidget(self)
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)

        if os.path.exists('./output/video_images') == False:
            os.mkdir('./output/video_images')

    def detroy_layout(self):
        if self.formLayout is not None:
            
            while self.formLayout.count():
                item = self.formLayout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

    def decode_option_changed(self):
        self.decode_option = self.comboBox.currentIndex()
        if self.decode_option == 0:
            self.lineEdit.setDisabled(True)
        else :
            self.lineEdit.setDisabled(False)
        
    def decode_btn_func(self):
        if os.path.exists(self.video) == False:
            QMessageBox.critical(self,'Error','Video file decode failed')
            return
        try:
            self.cap = cv2.VideoCapture(self.video)
        except:
            QMessageBox.critical(self,'Error','Video file decode failed')
            return
        
        if(self.decode_option == 1):
            value = float(self.lineEdit.text())
            if value <0 and value > 1:
                QMessageBox.critical(self,'Error','threshold not avarible')
                return
            self.threshold = value
        
        self.totalfs = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.decode_idx = 0
        success,self.first_frame = self.cap.read()
        self.first_frame = cv2.cvtColor(self.first_frame,cv2.COLOR_BGR2GRAY)
        self.dir_name = os.path.basename(self.video)
        self.dir_name = self.dir_name[0:self.dir_name.rfind('.')]
        path = './output/video_images/%s'%self.dir_name
        if os.path.exists(path) == False:
            os.mkdir(path)
        else:
            images = get_images(path)
            for img in images:
                os.remove(img)
        self.saved_count = 0
        self.nextButton.setDisabled(True)
        self.prevButton.setDisabled(True)
        self.Decode.setDisabled(True)
        self.Open.setDisabled(True)
        self.comboBox.setDisabled(True)
        if self.decode_option == 0:
            self.lineEdit.setDisabled(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timer_timeout_func)
        self.timer.start(30)
        self.update()
        
    def show_cv_image(self,cv_im):
        img = cv2.resize(cv_im,(self.image_w, self.image_h))
        img_height, img_width, channels = img.shape
        bytesPerLine = channels * self.image_w
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB,img)
        self.image = QImage(img.data,self.image_w, self.image_h,bytesPerLine,QImage.Format_RGB888)
        self.imagelabel.setPixmap(QPixmap.fromImage(self.image))

    def timer_timeout_func(self):
         
        state,frame = self.cap.read()
        if state:
            fname = './output/video_images/' + self.dir_name + "/%05d.jpg"%self.decode_idx
            self.decode_idx = self.decode_idx +1
            if self.decode_option == 0:  
                cv2.imwrite(fname,frame)
                self.show_cv_image(frame)
                self.saved_count = self.saved_count +1
            else:
                current = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                result = cv2.absdiff(self.first_frame,current)
                result = cv2.threshold(result,10,255.0,cv2.THRESH_BINARY)
                result_arr = np.array(result[1]).flatten()
                max_count = np.sum(result_arr == 255) / len(result_arr)
                if max_count >= self.threshold:
                    cv2.imwrite(fname,frame)
                    self.show_cv_image(frame)
                    self.first_frame = current
                    self.saved_count = self.saved_count +1

        else:
            self.infolabel.setText('Decode %d images to %s'%(self.saved_count,('./output/video_images/' + self.dir_name)))
            self.timer.stop()
            self.nextButton.setDisabled(False)
            self.prevButton.setDisabled(False)
            self.Decode.setDisabled(False)
            self.Open.setDisabled(False)
            self.comboBox.setDisabled(False)
            if self.decode_option == 1:
                self.lineEdit.setDisabled(False)
            self.decoded_images = get_images('./output/video_images/' + self.dir_name)
            self.decoded_images_idx = len(self.decoded_images) - 1
            self.show_image()
            
        self.progressBar.setValue(int((self.decode_idx +1) / self.totalfs * 100))
        self.update()
    
    def show_image(self):
        if len(self.decoded_images) > 0:
            img = cv2.imread(self.decoded_images[self.decoded_images_idx])
            img = cv2.resize(img,(self.image_w, self.image_h))
            img_height, img_width, channels = img.shape
            bytesPerLine = channels * self.image_w
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB,img)
            self.image = QImage(img.data,self.image_w, self.image_h,bytesPerLine,QImage.Format_RGB888)
            self.imagelabel.setPixmap(QPixmap.fromImage(self.image))

    def prev_btn_func(self):
        if(len(self.decoded_images) == 0):
            QMessageBox.critical(self,'Error','No decoded image')
            return 

        if self.decoded_images_idx > 0:
            self.decoded_images_idx = self.decoded_images_idx -1
        else :
            self.decoded_images_idx = len(self.decoded_images) - 1
        self.show_image()
        self.progressBar.setValue(int((self.decoded_images_idx +1) / len(self.decoded_images) * 100))

    def next_btn_func(self):
        if(len(self.decoded_images) == 0):
            QMessageBox.critical(self,'Error','No decoded image')
            return
        self.decoded_images_idx = (self.decoded_images_idx + 1 )%(len(self.decoded_images))
        self.show_image()
        self.progressBar.setValue(int((self.decoded_images_idx +1) / len(self.decoded_images) * 100))

    def open_btn_func(self):
        self.video,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Open mp4 video", '.', '(*.mp4)')
        print(self.video)
    def dialg_connect(self):
        print('diag exit')
    def setupUi(self,option=0):
        
        self.decode_option = option
        Dialog = self.diag
        self.imagelabel = QtWidgets.QLabel(Dialog)
        self.imagelabel.setGeometry(QtCore.QRect(0, 40, 480, 360))
        self.image_w = 480
        self.image_h = 360
        self.imagelabel.setObjectName("imagelabel")
        self.infolabel = QtWidgets.QLabel(Dialog)
        self.infolabel.setGeometry(QtCore.QRect(0, 0, 480, 40))
        self.Decode = QtWidgets.QPushButton(Dialog)
        self.Decode.setGeometry(QtCore.QRect(550, 200, 93, 28))
        self.Decode.setText("Decode")
        self.Decode.clicked.connect(self.decode_btn_func)
        self.prevButton = QtWidgets.QPushButton(Dialog)
        self.prevButton.setGeometry(QtCore.QRect(550, 280, 93, 28))
        self.prevButton.setText("prevButton")
        self.prevButton.clicked.connect(self.prev_btn_func)
        self.nextButton = QtWidgets.QPushButton(Dialog)
        self.nextButton.setGeometry(QtCore.QRect(550, 330, 93, 28))
        self.nextButton.setText("nextButton")
       
        self.nextButton.clicked.connect(self.next_btn_func)
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setGeometry(QtCore.QRect(0, 410, 511, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.Open = QtWidgets.QPushButton(Dialog)
        self.Open.setGeometry(QtCore.QRect(550, 70, 93, 28))
        self.Open.setText("Open")
        self.Open.clicked.connect(self.open_btn_func)

        self.formLayoutWidget.setGeometry(QtCore.QRect(550, 130, 180, 60))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.comboBox = QtWidgets.QComboBox(self.formLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem('Decode All Frames')
        self.comboBox.addItem("Decode Key Frames")
        self.comboBox.setCurrentIndex(option)
        self.comboBox.activated.connect(self.decode_option_changed)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.comboBox)
        
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        self.label.setText("Threshold")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit.setText("%.2f"%self.threshold)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        if self.decode_option == 0:
            self.lineEdit.setDisabled(True)
    
    def closeEvent(self, event):
        if(len(self.decoded_images) == 0):
            self.dialogSignel.emit(0,'')
        else:
            self.dialogSignel.emit(1,'./output/video_images/' + self.dir_name)
            
        self.close()

       
    