#!/usr/bin/env python
# -*- coding: utf-8 -*-

'Main module'

__author__ = 'KoalaZB'

import cv2
import numpy as np
import zbar
from PIL import Image 

def p2pDistance(P, Q):
    """
    Calculate the distance from point P to point Q

    Parameters:
        -P: an array contains coordinates x and y of point P
        -Q: an array contains coordinates x and y of point Q   
    """
    return int(np.math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]), 2)))

def updateCorner(point1,point2,base,index,M):
    """
    Find the farest distance from point1 to point2 and update the max value "base" then record the specific point in M 

    Parameters:
        -point1: an array contains coordinates x and y of point1
        -point2: an array contains coordinates x and y of point2 
        -base: a baseline distance array
        -index: the index of baseline array  
        -M: an array for recording the specific point 
    """
    tmp_dist = p2pDistance(point1,point2)
    #print("didtance:",tmp_dist)
    if tmp_dist>base[index]:
        base[index] = tmp_dist
        #print("point1:x and y",point1[0],"   ",point1[1])
        M[0] = point1[0]
        M[1] = point1[1]

def bandCorner_Finder(orientation, tmpList, outList):
    """
    Adjust the finderPattern based on the orientation

    Parameters:
        -orientation: the orientation of QR Code 
        -tmpList: an array contains four coordinates of finderPattern 
        -outList: an array used to record four coordinates of finderPattern 
    """
    M0,M1,M2,M3 = [],[],[],[]
    if orientation==QR_ORI_NORTH:
        M0 = tmpList[0]
        M1 = tmpList[1]
        M2 = tmpList[2]
        M3 = tmpList[3]
    elif orientation==QR_ORI_EAST:
        M0 = tmpList[1]
        M1 = tmpList[2]
        M2 = tmpList[3]
        M3 = tmpList[0]
    elif orientation==QR_ORI_SOUTH:
        M0 = tmpList[2]
        M1 = tmpList[3]
        M2 = tmpList[0]
        M3 = tmpList[1]
    elif orientation==QR_ORI_WEST:
        M0 = tmpList[3]
        M1 = tmpList[0]
        M2 = tmpList[1]
        M3 = tmpList[2]
    outList.append(M0)
    outList.append(M1)
    outList.append(M2)
    outList.append(M3)

def dot2LineDistance(vertex1, vertex2, vertex):
    """
    Calculate the distance from point vertex to the line consisted of vertex1 and vertex2 

    Parameters:
        -vertex1: an array contains coordinates x and y of point vertex1
        -vertex2: an array contains coordinates x and y of point vertex2
        -vertex: an array contains coordinates x and y of point vertex   
    """
    a = -((vertex2[1]-vertex1[1])/(vertex2[0]-vertex1[0]))
    b=1.0
    c=(((vertex2[1]-vertex1[1])/(vertex2[0]-vertex1[0]))*vertex1[0])-vertex1[1]
    distance = (a*vertex[0]+(b*vertex[1])+c)/np.math.sqrt((a*a)+(b*b))
    return distance

def lineSlope(vertex1,vertex2,align):
    """
    Calculate the slope of line consisted of vertex1 and vertex2 

    Parameters:
        -vertex1: an array contains coordinates x and y of point vertex1
        -vertex2: an array contains coordinates x and y of point vertex2
        -align: a flag indicates the line is vertical   
    """
    dx = vertex2[0] - vertex1[0]
    dy = vertex2[1] - vertex1[1]
    if dy !=0:
        align[0] = 1;
        return (dy/dx)
    else:
        align[0] = 0;
        return 0.0
def getVertices(contours,cont_index,slope,tmpPoint):
    """
    Get the four vertices of current finderPattern

    Parameters:
        -contours: the contour arrays of the image
        -cont_index: the current finderPattern index in array contours
        -slope: the current slope of longest line 
        -tmpPoint: an array to store four vertices of  current finderPattern
    """
    A,B,C,D,W,X,Y,Z=[],[],[],[],[],[],[],[] #存放外接矩形坐标&四个中点坐标
    M0,M1,M2,M3=[0,0],[0,0],[0,0],[0,0] #存放四个角点坐标
    #获取轮廓的外接矩形(x:左上角横坐标，y:左上角纵坐标，w:矩形宽，h:矩形高）
    x,y,w,h = cv2.boundingRect(contours[cont_index])
    A.append(x)
    A.append(y)
    B.append(x+w)
    B.append(y)
    C.append(x+w)
    C.append(y+h)
    D.append(x)
    D.append(y+h)

    W.append(x+w/2)
    W.append(y)
    X.append(x+w)
    X.append(y+h/2)
    Y.append(x+w/2)
    Y.append(y+h)
    Z.append(x)
    Z.append(y+h/2)
    
    #print("  ",A,"  ",B,"  ",C,"  ",D,"  ")
    #print("  ",W,"  ",X,"  ",Y,"  ",Z,"  ")
    dist_max=[0.0,0.0,0.0,0.0]

    if slope>5 or slope<-5:
        for i in range(len(contours[cont_index])):
            pd1 = dot2LineDistance(C,A,contours[cont_index][i][0])
            pd2 = dot2LineDistance(B,D,contours[cont_index][i][0])
            if pd1>=0.0 and pd2>0.0:
                updateCorner(contours[cont_index][i][0],W,dist_max,1,M1)
            elif pd1>0.0 and pd2<=0.0:
                updateCorner(contours[cont_index][i][0],X,dist_max,2,M2)
            elif pd1<=0.0 and pd2<0.0:
                updateCorner(contours[cont_index][i][0],Y,dist_max,3,M3)
            elif pd1<0.0 and pd2>=0.0:
                updateCorner(contours[cont_index][i][0],Z,dist_max,0,M0)
            else:
                continue
    else:
    	#print("else_slope!!")
        midX = (A[0]+B[0])/2
        midY = (A[1]+D[1])/2
        #print("midX=",midX," midY=",midY)
        for i in range(len(contours[cont_index])):
            tmpX = contours[cont_index][i][0][0]
            tmpY = contours[cont_index][i][0][1]
            #print("x=",tmpX," y=",tmpY)
            if tmpX<midX and tmpY<=midY:
                updateCorner(contours[cont_index][i][0],C,dist_max,2,M0)
            elif tmpX>=midX and tmpY<midY:
                updateCorner(contours[cont_index][i][0],D,dist_max,3,M1)
            elif tmpX>midX and tmpY>=midY:
                updateCorner(contours[cont_index][i][0],A,dist_max,0,M2)
            elif tmpX<=midX and tmpY>midY:
            	#print ("hello m3")
                updateCorner(contours[cont_index][i][0],B,dist_max,1,M3)
    tmpPoint.append(M0)
    tmpPoint.append(M1)
    tmpPoint.append(M2)
    tmpPoint.append(M3)

def cross_ofPoints(P,Q):
    """
    Get the cross of two points

    Parameters:
        -P: an array contains coordinates x and y of point P
        -Q: an array contains coordinates x and y of point Q   
        
    Returns:
        return the cross of point P and Q
    """
    return P[0]*Q[1] - P[1]*Q[0]
    
def getIntersectionPoint(a1,a2,b1,b2):
    """
    Get the intersection of two lines which line1 formed by point a1 and a2,line2 formed by b1 and b2

    Parameters:
        -a1: an array contains coordinates x and y of point a1
        -a2: an array contains coordinates x and y of point a2   
        -b1: an array contains coordinates x and y of point b1
        -b2: an array contains coordinates x and y of point b2   
    Returns:
        return the intersection of two lines formed by point a1,a2 and b1,b2
    """
    r = [0,0]
    s = [0,0]
    b1_sub_a1 = [0,0]
    intersectionPoint = []

    b1_sub_a1[0] = b1[0]-a1[0]
    b1_sub_a1[1] = b1[1]-a1[1]

    r[0] = a2[0]-a1[0]
    r[1] = a2[1]-a1[1]

    s[0] = b2[0]-b1[0]
    s[1] = b2[1]-b1[1]

    if cross_ofPoints(r,s)==0:
        return intersectionPoint
    div_Cross = float(cross_ofPoints(b1_sub_a1,s))/cross_ofPoints(r,s)

    #print("div_Cross=",div_Cross)
    intersectionPoint.append(a1[0]+div_Cross*r[0])
    intersectionPoint.append(a1[1]+div_Cross*r[1])
    return intersectionPoint

def detect_QRCode():
    video_capture = cv2.VideoCapture(0)

    while True:
        QR_ORI_NORTH,QR_ORI_EAST,QR_ORI_SOUTH,QR_ORI_WEST = 0,1,2,3
        mark,A,B,C,top,right,bottom,vertex1,vertex2,vertex=0,0,0,0,0,0,0,0,0,0
        AB,BC,CA,dist,slope,areat,arear,areab,large,padding=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        orientation = 0
        align=[0]
        result = np.zeros((300,300),np.uint8)
        
        ret, frame = video_capture.read()
   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #用高斯滤波器平滑图像去除噪声干扰
        gaussianBlur = cv2.GaussianBlur(gray, (5, 5), 0)
        #bi_img = getBinaryImg(gaussianBlur)
        #利用canny算子进行边缘提取
        edges = cv2.Canny(gaussianBlur, 100, 200)
        #调用findContours查找图像轮廓和层级
        img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        mark = 0
        len_con = len(contours)
        mu = []
        mc = []
        #获取所有轮廓的moment（矩）以及中心
        for i in range(len_con):
            tmp=[]
            mu.append(cv2.moments(contours[i]))
            tmpdiv=mu[i]['m00']
            if tmpdiv==0.0:
                #print i
                tmpdiv=1
            cx = mu[i]['m10']/tmpdiv
            cy = mu[i]['m01']/tmpdiv
            tmp.append(cx)
            tmp.append(cy)
            mc.append(tmp)

        for i in range(len_con):
            epsilon = 0.02*cv2.arcLength(contours[i],True)
            approx = cv2.approxPolyDP(contours[i],epsilon,True)
            if len(approx)==4:
                j = i
                c = 0
                while hierarchy[j][2] != -1:
                    j = hierarchy[j][2]
                    c = c + 1  # 记录轮廓层级
                if c >= 5:
                    if mark==0:
                        A = i
                    elif mark==1:
                        B = i
                    elif mark==2:
                        C = i
                    mark = mark + 1
        if mark==3: 
                #计算三个定位图案两两距离
                AB = p2pDistance(mc[A],mc[B])
                BC = p2pDistance(mc[B],mc[C])
                CA = p2pDistance(mc[C],mc[A])
                #判断顶点定位图案
                if AB>BC and AB>CA:
                    vertex = C
                    vertex1 = A
                    vertex2 = B
                elif CA>AB and CA>BC:
                    vertex = B
                    vertex1 = A
                    vertex2 = C
                elif BC>AB and BC>CA:
                    vertex = A
                    vertex1 = B
                    vertex2 = C
                top = vertex
                #确定当前QR Code的方位，以及三个定位图案的对应关系（top,right,bottom)
                dist = dot2LineDistance(mc[vertex1], mc[vertex2], mc[vertex])
                slope = lineSlope(mc[vertex1],mc[vertex2],align)

                if align[0]==0: 
                    bottom = vertex1
                    right = vertex2
                elif slope<0 and dist<0: #朝向：北
                    bottom = vertex1
                    right = vertex2
                    orientation = QR_ORI_NORTH
                elif slope>0 and dist<0: #朝向：东
                    right = vertex1
                    bottom = vertex2
                    orientation = QR_ORI_EAST
                elif slope<0 and dist>0: #朝向：南 
                    right = vertex1
                    bottom = vertex2
                    orientation = QR_ORI_SOUTH
                elif slope>0 and dist>0: #朝向：西 
                    bottom = vertex1
                    ight = vertex2
                    orientation = QR_ORI_WEST

                print("orientation: ",orientation)

                topArea = cv2.contourArea(contours[top])
                rightArea = cv2.contourArea(contours[right])
                bottomArea = cv2.contourArea(contours[bottom])

                if top<len_con and right<len_con and bottom<len_con and topArea>10 and rightArea>10 and bottomArea>10:
                    L,M,O,tmpL,tmpM,tmpO=[],[],[],[],[],[]
                    #获取所有定位图案的四个顶点坐标，并推算出QR Code的第四个角点位置
                    getVertices(contours,top,slope,tmpL)
                    getVertices(contours,right,slope,tmpM)
                    getVertices(contours,bottom,slope,tmpO)

                    #print ("tmpL :",tmpL)
                    #print ("tmpM :",tmpM)
                    #print ("tmpO :",tmpO)
                    bandCorner_Finder(orientation,tmpL,L)
                    bandCorner_Finder(orientation,tmpM,M)
                    bandCorner_Finder(orientation,tmpO,O)

                    #print ("L :",L)
                    #print ("M :",M)
                    #print ("O :",O)
                    #print("M[1]=",M[1]," M[2]=",M[2]," O[3]=",O[3]," O[2]=",O[2])
                    #第四个角点位置
                    intersectionPoint = getIntersectionPoint(M[1],M[2],O[3],O[2])
                    
                    #通过形态学进行透视变换，规范QR Code 的位置
                    src = np.float32([L[0],M[1],intersectionPoint,O[3]])

                    #print("src",src," ",len(src))
                    dst = np.float32([[0,0],[300,0],[300,300],[0,300]])
                    #print("dst",dst," ",len(dst))
                    if len(src) == 4 and len(dst)==4:
                        M_perspective = cv2.getPerspectiveTransform(src,dst)
                        img_perspective = cv2.warpPerspective(frame, M_perspective, (300, 300))
                        result = cv2.copyMakeBorder(img_perspective, 30 , 30, 30, 30, cv2.BORDER_CONSTANT, (0,0,255))
                        datas = zbarDecodeQR(frame)
                        font=cv2.FONT_HERSHEY_SIMPLEX
                        result = cv2.putText(result,datas,(10,40),font,0.5,(255,255,255),1)
        cv2.imshow("Frame",frame)
        cv2.imshow("Result",result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
    #print("ave1=",ave1," ave2=",ave2," ave3=",ave3," ave4=",ave4)
    threshold1,threshold2,threshold3,threshold4 = ave1*0.94,ave2*0.94,ave3*0.94,ave4*0.94
    #print("threshold1=",threshold1," threshold2=",threshold2," threshold3=",threshold3," threshold4=",threshold4)
    ret1, bi_img1 = cv2.threshold(block1, threshold1, 255, cv2.THRESH_BINARY) 
    ret2, bi_img2 = cv2.threshold(block2, threshold2, 255, cv2.THRESH_BINARY) 
    ret3, bi_img3 = cv2.threshold(block3, threshold3, 255, cv2.THRESH_BINARY) 
    ret4, bi_img4 = cv2.threshold(block4, threshold4, 255, cv2.THRESH_BINARY) 

    result_img = np.hstack([np.vstack([bi_img1,bi_img2]),np.vstack([bi_img3,bi_img4])])
    
    
    kernel1=np.array([[0,0,0],[1,1,1],[0,0,0]],np.uint8)
    kernel2=np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)

    erosion=cv2.erode(result_img,kernel1,iterations=1)
    dilation=cv2.dilate(erosion,kernel2,iterations=1)
    result_img = dilation
    return result_img

def zbarDecodeQR(finalQR):
    image = Image.fromarray(finalQR).convert('L')
    img_scan = zbar.ImageScanner()

    img_scan.parse_config('enable')
    width, height = image.size

    qrCode = zbar.Image(width, height, 'Y800', image.tobytes())
    img_scan.scan(qrCode)

    data = ''
    for s in qrCode:
        data += s.data
 
    del image
    return data

if __name__ == '__main__':
    QR_ORI_NORTH,QR_ORI_EAST,QR_ORI_SOUTH,QR_ORI_WEST = 0,1,2,3
    detect_QRCode()	
    cv2.waitKey (0)  
    cv2.destroyAllWindows() 