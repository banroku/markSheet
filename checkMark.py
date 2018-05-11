#!/usr/bin/python3
import os
import numpy as np
import cv2
from pdf2image import convert_from_path

def readPDF(infile, width, grayscale=True):
    """pdfを読み込み、opevCVで使えるarrayに変換。{{{
    pdfが複数ページの場合は、1ページ目を保存する仕様。
    """ 
    # pdfを読み込む

    imgAllPages = convert_from_path(infile, dpi=100)
    img = imgAllPages[0]  #pick first page up
    img = np.asarray(img)
    img = img.take([1,2,0], axis=2)  #channel変換(GBR -> RGB)
    
    # 幅が1084からずれていた場合、拡大or縮小する

    if img.shape[1] != width:
        height = int(round(img.shape[0] / img.shape[1] * width))
        img = cv2.resize(img, (width, height), 
                         interpolation = cv2.INTER_CUBIC)

    #グレイスケールに変換

    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img  
    #}}}

def correctMisalign(img, marker, center, compus, scope=100):
    """画像の位置ズレを調整 {{{
    並進ズレをcenterに揃え、回転ズレをcenter-compusに揃える
    img: 調整する画像
    marker: 目印となる画像(円)。中心が画像の中央にくること。
    center: [x, y]
    compus: [x, y]
    scope: 本来の目印からのズレの許容範囲
    """

    markerCenter = np.asarray(marker.shape)//2
    guide = np.asarray([center, compus])
    landmark = np.zeros(guide.shape)
    
    #template matching 1回目 (ズレ修正のためのlandmarkの読取り用) 
    result = cv2.matchTemplate(img, marker, 0)
    result = (1-result/np.max(result))*255
    M = np.float32([
    [1, 0, markerCenter[1]] ,
    [0, 1, markerCenter[0]] ])
    resultPadded = cv2.warpAffine(result, M, (width, height))
    
    mask = np.zeros(resultPadded.shape)

    for i in range(0, len(guide)):
        mask[:] = 0
        mask_xfr = max(0, guide[i,1]-(scope+markerCenter[0]))
        mask_xto = min(width, guide[i,1]+(scope+markerCenter[0]))
        mask_yfr = max(0, guide[i,0]-(scope+markerCenter[1]))
        mask_yto = min(width, guide[i,0]+(scope+markerCenter[1]))
        mask[mask_xfr:mask_xto, mask_yfr:mask_yto] = 255
        min_val, max_val, min_val, landmark[i,:] = cv2.minMaxLoc(np.multiply(resultPadded, mask))
    
    #ズレ修正1 (shift)
    shift = guide[0] - landmark[0] 
    M = np.float32([
    [1, 0, shift[0]] ,
    [0, 1, shift[1]] ])
    imgShifted = cv2.warpAffine(img, M, (width, height))
    
    #ズレ修正2 (scale & rotate)
    radius = np.linalg.norm(landmark[1,:] - landmark[0,:])
    scale = np.linalg.norm(guide[1,:] - guide[0,:])/radius
    cos = (landmark[1,0]-landmark[0,0])/radius
    theta = np.arccos(cos) / (2 * np.pi) * 360
    M = cv2.getRotationMatrix2D((guide[0,0],guide[0,1]),-theta,scale)
    imgModified = cv2.warpAffine(imgShifted,M,(width,height))
    return imgModified

    #}}}

def checkAnswer(img, marker, answerList, threshold=110):
    """ checkAnswer checks answers according to #{{{
    the coordinate of answerList.
    answerList = [['answer name', x, y, '0/1'], ..., ]
    """

    markerCenter = np.asarray(marker.shape)//2
    width = img.shape[1]
    height = img.shape[0]

    #template matching 2回目 (回答読み取り用)
    resultFinal = cv2.matchTemplate(imgModified, marker, 0)
    resultFinal = (1-resultFinal/np.max(resultFinal))*255
    M = np.float32([
    [1, 0, markerCenter[1]] ,
    [0, 1, markerCenter[0]] ])
    resultFinal = cv2.warpAffine(resultFinal, M, (width, height))

    #answerList のlistから、x,y座標だけを取り込む
    answerCoord = np.asarray(answerList)
    answerCoord = np.asarray(answerCoord[:,1:3], dtype=np.int)
    
    for i in range(0, answerCoord.shape[0]):
        if (resultFinal[answerCoord[i,1], answerCoord[i,0]] 
            > threshold):
            answerList[i][3] = 1
        else:
            answerList[i][3] = 0
        print(answerList[i][0], ':', answerList[i][3])

    return answerList
    #}}}

if __name__ == '__main__':
    """ scanImage/フォルダ内のすべてのファイルに対して、
    回答のチェックを実施する
    """
    
    #center/compus座標の設定
    width = 1084
    center = [31,  726]
    compus = [1051, 726]
    
    #回答読み取り座標の設定
    answerList = [
        ['category1', 951, 42, 0],
        ['category2', 990, 42, 0],
        ['category3', 1030, 42, 0],
        ['molding', 115, 668, 0],
        ['processing', 223, 668, 0],
        ['evaluation', 369, 668, 0],
        ['data-analysis', 512, 668, 0], 
        ['discussion', 680, 668, 0],
        ['formula', 800, 668, 0],
        ['synthesis', 950, 668, 0],
        ['CAE', 115, 708, 0], 
        ['others', 223, 708, 0],
    ]

    #フォルダ内のファイルのリストを取得
    files = os.listdir('./scanPDF')
    files_file = [f for f in files if os.path.isfile(os.path.join('./scanPDF', f))]
    
    for i in range(0, len(files_file)):
        infile = os.path.join('./scanPDF', files_file[i])
        img = readPDF(infile, width)
        
        width = img.shape[1]
        height = img.shape[0]
        
        #目印画像の設定
        marker= cv2.imread('landmark.png', 0)
        imgModified = correctMisalign(img, marker, center, compus, scope = 100)
        print(infile)
        checkAnswer(imgModified, marker, answerList, threshold=110)

# vim: set foldmethod=marker :
