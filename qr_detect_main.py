#!/usr/bin/env python
# -*- coding: utf-8 -*-

'Main module'

__author__ = 'KoalaZB'
import time
import cv2
import numpy as np
from PIL import Image 
from PIL import ImageEnhance

import img_basic_operation
import img_operation
import common_func

def detect_QRcode(src_img_path):
    '''
    --------------------图像初始化操作-----------------------
    step1:图像采样
    step2:灰度化处理
    step3:增强对比度
    '''	
    #读取原始待处理图像-----采样
    src_img = cv2.imread(src_img_path)
    #cv2.imshow("src_img",src_img)

    t = time.time()
    start = round(time.time()*1000)     
    #图像灰度化处理
    src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("src_img_gray", src_img_gray) 
    cv2.imwrite("process/src_img_gray.jpg",src_img_gray) 
    #common_func.draw_histo(src_img_gray.flatten(),"src_gray_histograph")
    print u"图像灰度化处理完成！"
    contrast = img_basic_operation.getAveSqrt(src_img_gray)
    #contrast = 18
    #print("contrast is ",contrast)
    #图像直方图均衡化/对比度拉伸操作（增强对比度）
    if contrast<18:
        dst_img_init = img_basic_operation.img_Histo_Stretch(src_img_gray.copy())
        #dst_img_init1 =  img_basic_operation.img_Histo_Equalize(dst_img_init1.copy())
        #common_func.draw_histo(dst_img_iimg_basic_operationnit.flatten(),"dst_img_init_histograph")
        #cv2.imwrite("dst_img_init.jpg",dst_img_init)
       
    else:
        dst_img_init = src_img_gray.copy()
    cv2.imshow("initialized_img",dst_img_init)
    cv2.imwrite("process/dst_img_init.jpg",dst_img_init) 
    print u"图像对比度调整完成！"
    '''
    --------------------图像降噪&边缘提取-----------------------
    --------------------图像二维码定位操作-----------------------
    #step1:图像减噪
    #step2:图像边缘提取
    #step3:QR Code locate
    #step4:QR Code 信息提取
    '''
    locatedQR = img_operation.qrcode_detection(src_img_gray,src_img_path,contrast)
    end = round(time.time()*1000)    
    print ("time=",end-start)

if __name__ == '__main__':
    #8 15
    src_img_path = "F:\ZB‘s Files\TsengBiao\毕业设计\result\Code\QRcode_final\QRcode\imgs\21.jpg"
    detect_QRcode(src_img_path)	
    cv2.waitKey (0)  
    cv2.destroyAllWindows() 