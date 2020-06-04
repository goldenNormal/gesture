# coding: utf-8
import numpy as np
import cv2
import time
import os

minValue = 70
#样本数目
sample_nums = 0
#保存样本变量
counter = 0
#手势名称
gestname = ""
#样本路径
path = ""
#保存图片开关
saveimg = False
#识别框初始值
x0 = 0
y0 = 280
height = 200
width = 200
head=cv2.imread('head.png')

#保存图片函数
def saveROIImg(img):
    global counter,saveimg,gestname,sample_nums
    if counter > sample_nums:
        saveimg = False
        counter = 0
        gestname = ""
        return 
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".png", img)
    #防止保存图片过快，造成手势无变化
    time.sleep(0.04 )
    

#手势处理函数(二值掩模)
def binaryMask(frame, x0, y0, width, height):
    #只处理识别框部分
    # print(x0,y0)
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if saveimg==True:
        saveROIImg(res)
    return res

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
        framecount = framecount + 1
        #结束时间
        end  = time.time()
        second = (end - start)
        if( second >= 1):
            #计算帧率
            fps = 'FPS:%s' %(framecount)
            start = time.time()
            framecount = 0
    #fps值
    cv2.putText(frame,fps,(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,1)
    #相关操作提示(按C键来创建一个手势文件夹，按S键开始收集手势数据集

    #遮挡头像
    face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.1, 2)
    for x, y, w, h in faces:
        frame[y:y+h,x:x+w]=cv2.resize(head,(w,h))

    #
    cv2.putText(frame,'push key c to create floder',(10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,1)
    cv2.putText(frame,'push key s to save samples',(10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,1)
    
    #显示摄像头内容和处理后手势的图像内容
    cv2.imshow('Original',frame)
    cv2.imshow('ROI', roi)


    key = cv2.waitKey(5) & 0xff
    #Esc键退出
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
    elif key == ord('c'):
        gestname = input("输入存放手势的文件夹名称: ")
        sample_nums = int(input("输入存放手势图片数目: "))
        try:
            os.makedirs(gestname)
        except OSError as e:
            print(gestname+'文件夹已创建')
        path = "./"+gestname+"/"
    elif key == ord('s'):
        if gestname=='':
            print("请先输入一个存放文件夹的名字")
        else:
            saveimg = True
    #手势识别框动态移动
    elif key == ord('i'):
        y0 = y0 - 5
    elif key == ord('k'):
        y0 = y0 + 5
    elif key == ord('j'):
        x0 = x0 - 5
    elif key == ord('l'):
        x0 = x0 + 5
        
        

