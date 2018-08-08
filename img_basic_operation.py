#!/usr/bin/env/ python
#encoding=UTF-8

'Image Contrast  Ratio Enhancement'

__author__ = 'KoalaZB'

import cv2
import numpy as np

'''
    直方图均衡化，增强图像对比度

    Parameters:
        -src_img_gray: a cv2 image instance src_img_gray

    Returns:
        -result_image: a cv2 image instance of histo-euqlized
'''
def img_Histo_Equalize(src_img_gray):

	#计算图像直方图
	hist,bins = np.histogram(src_img_gray.flatten(),256,[0,256]) 
	
	 #计算累积直方图  
	cumulate_histo = hist.cumsum() 

	#除去直方图中的0值  
	ch_m = np.ma.masked_equal(cumulate_histo,0)  
	ch_m = (ch_m - ch_m.min())*255/(ch_m.max()-ch_m.min()) 

	#将掩模处理掉的元素补为0  
	cumulate_histo = np.ma.filled(ch_m,0).astype('uint8')  
	  
	#计算直方图均衡后的图像
	result_image = cv2.LUT(src_img_gray, cumulate_histo) 
	return result_image

'''
    直方图对比度拉伸，增强图像对比度

    Parameters:
        -src_img_gray: a cv2 image instance of gray 

    Returns:
        -result_image: a cv2 image instance of histo-strentched
'''
def img_Histo_Stretch(src_img_gray):
    #创建空的查找表  
    finding_table = np.zeros(256, dtype = src_img_gray.dtype )
    hist= cv2.calcHist([src_img_gray], [0], None, [256], [0.0,255.0])  #范围
          
    minBinNo, maxBinNo = 0, 255  
      
    #查找从左起第一个不为0的直方图柱的位置  
    for binNo, binValue in enumerate(hist):  
        if binValue != 0:  
            minBinNo = binNo  
            break  
    #查找从右起第一个不为0的直方图柱的位置  
    for binNo, binValue in enumerate(reversed(hist)):  
        if binValue != 0:  
            maxBinNo = 255-binNo  
            break  
      
    #生成查找表
    for i,v in enumerate(finding_table):  
        if i < minBinNo:  
            finding_table[i] = 0  
        elif i > maxBinNo:  
            finding_table[i] = 255  
        else:  
            finding_table[i] = int(255.0*(i-minBinNo)/(maxBinNo-minBinNo)+0.5)  
      
    #通过查找表生成新的图像 
    result_image = cv2.LUT(src_img_gray, finding_table)  
    return result_image

'''
    计算4X4子块最大值减去最小值的结果

    Parameters:
        -tmp: a 4x4 array block 

    Returns:
        -maxValue - minValue: substration result of maxValue and minValue
'''
def max_sub_min(tmp):
    maxValue = 0
    minValue = 0
    for i in range(len(tmp[0])):
        mValue = max(tmp[i])
        nValue = min(tmp[i])
        if mValue>maxValue:
            maxValue = mValue
        if nValue<minValue:
            minValue = nValue
    return maxValue - minValue

'''
    Max-Min差分减少图像背景噪声

    Parameters:
        -img: a cv2 image instance 

    Returns:
        -img: a cv2 image instance after max-min differencial
'''
def Max_Min(img):
    row = img.shape[0]
    col = img.shape[1]

    print ("row:",row,"col",col)
    for i in range(int(row/4)):
        for j in range(int(col/4)):
            tmp = img[i*4:(i+1)*4,j*4:(j+1)*4]
            img[i*4][j*4] = max_sub_min(tmp)

    return img

'''
    计算子块像素平均值

    Parameters:
        -block: an array block
        -aveSize: total pixels of current block

    Returns:
        -sumPixel/aveSize: average pixel of current block
'''
def getAvePixel(block,aveSize):
    sumPixel = 0
    for i in range(len(block)):
        for j in range(len(block[0])):
            sumPixel = sumPixel + block[i][j]
    return sumPixel/aveSize

'''
    对图像进行二值化

    Parameters:
        -image: a cv2 image instance

    Returns:
        -result_img: a cv2 image instance after binarized
'''
def getBinaryImg(image):
    row = image.shape[0]
    col = image.shape[1]
    if row<800 or col<800:
        ret1, final_bi_img = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    else:
        #将图像分为四块
        block1 = image[0:row/2,0:col/2]
        block2 = image[row/2:row,0:col/2]
        block3 = image[0:row/2,col/2:col]
        block4 = image[row/2:row,col/2:col]

        #计算每块的像素平均值
        aveSize = row*col/4
        ave1 = getAvePixel(block1,aveSize)
        ave2 = getAvePixel(block2,aveSize)
        ave3 = getAvePixel(block3,aveSize)
        ave4 = getAvePixel(block4,aveSize)
        print("ave1=",ave1," ave2=",ave2," ave3=",ave3," ave4=",ave4)
        threshold1,threshold2,threshold3,threshold4 = ave1*0.94,ave2*0.94,ave3*0.94,ave4*0.94
        print("threshold1=",threshold1," threshold2=",threshold2," threshold3=",threshold3," threshold4=",threshold4)
        ret1, bi_img1 = cv2.threshold(block1, threshold1, 255, cv2.THRESH_BINARY) 
        ret2, bi_img2 = cv2.threshold(block2, threshold2, 255, cv2.THRESH_BINARY) 
        ret3, bi_img3 = cv2.threshold(block3, threshold3, 255, cv2.THRESH_BINARY) 
        ret4, bi_img4 = cv2.threshold(block4, threshold4, 255, cv2.THRESH_BINARY) 

        #ret1, bi_img1 = cv2.threshold(block1, 0, 255, cv2.THRESH_OTSU)
        #ret2, bi_img2 = cv2.threshold(block1, 0, 255, cv2.THRESH_OTSU) 
        #ret3, bi_img3 = cv2.threshold(block2, 0, 255, cv2.THRESH_OTSU) 
        #ret4, bi_img4 = cv2.threshold(block3, 0, 255, cv2.THRESH_OTSU) 

        final_bi_img = np.hstack([np.vstack([bi_img1,bi_img2]),np.vstack([bi_img3,bi_img4])])
    return final_bi_img
'''
    获取图像对比度
    Parameters:
        -image: a cv2 image instance

    Returns:
        -contrast: the contrast of current image
'''
def getAveSqrt(image):
    row = image.shape[0]
    col = image.shape[1]
    sumPixel = 0
    for i in range(row/4):
        for j in range(col/4):
            sumPixel = sumPixel + image[i][j]
    avePixel = sumPixel/(row*col)

    deltaPixel = 0
    for i in range(row/4):
        for j in range(col/4):
            deltaPixel = deltaPixel + pow((image[i][j]-avePixel),2)
    sum_all = []
    sum_all.append(deltaPixel/(row*col))
    contrast = np.sqrt(sum_all)[0]
    return contrast