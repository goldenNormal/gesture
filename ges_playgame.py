# coding: utf-8

from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import time
import os
import win32api
import win32con

minValue = 70
x0 = 0
y0 = 280
height = 200
width = 200

def keybd_event(VK_CODE):
    #VK_CODE为键盘编码
    VK_CODE = int(VK_CODE)
    #按键按下
    win32api.keybd_event(VK_CODE, 0, 0, 0)
    #按键弹起
    win32api.keybd_event(VK_CODE, 0, win32con.KEYEVENTF_KEYUP, 0)

#手势处理函数(二值掩模)
def binaryMask(frame, x0, y0, width, height):
    global guessGesture, visualize, mod, lastgesture, saveImg
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res


#模型和标签名
MODEL_NAME = "ss_model.h5"
LABEL_NAME = "ss_labels.dat"


#加载标签
with open(LABEL_NAME, "rb") as f:
    lb = pickle.load(f)

#加载神经网络
model = load_model(MODEL_NAME)


#打开摄像头
cap = cv2.VideoCapture(1)

framecount = 0
fps = ""
#开始时间
start = time.time()
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 3)
    frame = cv2.resize(frame, (640,480))   

    if ret == True:
        roi = binaryMask(frame, x0, y0, width, height)
        roi1 = cv2.resize(roi,(100,100))
        # 添加维度
        roi1 = np.expand_dims(roi1, axis=2)
        roi1 = np.expand_dims(roi1, axis=0)
        prediction = model.predict(roi1)
        # 预测手势
        gesture = lb.inverse_transform(prediction)[0]
        cv2.putText(frame,gesture,(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        #计算帧率
        framecount = framecount + 1
        end  = time.time()
        second = (end - start)
        if( second >= 1):
            fps = 'FPS:%s' %(framecount)
            start = time.time()
            framecount = 0
    #输出fps
    cv2.putText(frame,fps,(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,1)

    #显示摄像头内容和处理后手势的图像内容
    cv2.imshow('Original',frame)
    cv2.imshow('ROI', roi)

    if gesture=='left':
        keybd_event(37) #键盘按下左

    elif gesture=='right':
        keybd_event(39) #键盘按下右

    elif gesture=='up':
        keybd_event(38)#键盘按上

    else:
        win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(37, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(39, 0, win32con.KEYEVENTF_KEYUP, 0)



    key = cv2.waitKey(3) & 0xff
    #Esc键退出
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
    #手势识别框动态移动
    elif key == ord('i'):
        y0 = y0 - 5
    elif key == ord('k'):
        y0 = y0 + 5
    elif key == ord('j'):
        x0 = x0 - 5
    elif key == ord('l'):
        x0 = x0 + 5
