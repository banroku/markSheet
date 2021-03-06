# checkMark.py checks marks on the answer sheet. 

import os
import numpy as np
import cv2
from pdf2image import convert_from_path

landmark = np.zeros((3,2))

#フォルダ内のファイルのリストを取得
files = os.listdir('./scanPDF')
files_file = [f for f in files if os.path.isfile(os.path.join('./scanPDF', f))]

infile = os.path.join('./scanPDF', files_file[0])

#pdfを読み込み、opevCVで使えるarrayに変換
imgAll = convert_from_path(infile, dpi=100)
img = imgAll[0]    #pick first page up
img = np.asarray(img)
img = img.take([1,2,0], axis=2) #channel変換(GBR -> RGB)

#幅が1084からずれていた場合、拡大or縮小する
if img.shape[1] != 1084:
    width = 1084
    height = int(round(img.shape[0] / img.shape[1] * 1084))
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)

#グレイスケールに変換
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#marker画像の登録
marker= cv2.imread('landmark.png', 0)
markerCenter = np.array([20, 21])

#template matching 1回目 (ズレ修正のためのlandmarkの読取り用) 
result = cv2.matchTemplate(img, marker, 0)
result = (1-result/np.max(result))*255
M = np.float32([
[1, 0, markerCenter[1]] ,
[0, 1, markerCenter[0]] ])
resultPadded = cv2.warpAffine(result, M, (width, height))

guide = np.array([
[726, 31],
[726, 1051],
[29, 31], ])

#landmark座標の読み取り
scope = 100  #本来のguide点からのズレの許容範囲
mask = np.zeros(resultPadded.shape)

for i in range(0, landmark.shape[0]):
    mask[:] = 0
    mask_xfr = max(0, guide[i,0]-(scope+markerCenter[0]))
    mask_xto = min(width, guide[i,0]+(scope+markerCenter[0]))
    mask_yfr = max(0, guide[i,1]-(scope+markerCenter[1]))
    mask_yto = min(width, guide[i,1]+(scope+markerCenter[1]))
    mask[mask_xfr:mask_xto, mask_yfr:mask_yto] = 255
    min_val, max_val, min_val, landmark[i,:] = cv2.minMaxLoc(np.multiply(resultPadded, mask))

landmark = np.take(landmark, [1,0], axis=1) #x,yが逆転しているのを直す

#ズレ修正1 (shift)
shift = guide[0] - landmark[0] 
M = np.float32([
[1, 0, shift[1]] ,
[0, 1, shift[0]] ])
imgShifted = cv2.warpAffine(img, M, (width, height))

#ズレ修正2 (scale & rotate)
radius = np.linalg.norm(landmark[1,:] - landmark[0,:])
scale = np.linalg.norm(guide[1,:] - guide[0,:])/radius
cos = (landmark[1,1]-landmark[0,1])/radius
theta = np.arccos(cos) / (2 * np.pi) * 360
M = cv2.getRotationMatrix2D((guide[0,1],guide[0,0]),-theta,scale)
imgModified = cv2.warpAffine(imgShifted,M,(width,height))

#template matching 2回目 (回答読み取り用)
resultFinal = cv2.matchTemplate(imgModified, marker, 0)
resultFinal = (1-resultFinal/np.max(resultFinal))*255
M = np.float32([
[1, 0, markerCenter[1]] ,
[0, 1, markerCenter[0]] ])
resultFinal = cv2.warpAffine(resultFinal, M, (width, height))

#judge answer 
answer = [
['category1', 42, 951, 0],
['category2', 42, 990, 0],
['category3', 42, 1030, 0],
['molding', 668, 115, 0],
['processing', 668, 223, 0],
['evaluation', 668, 369, 0],
['data-analysis', 668, 512, 0], 
['discussion', 668, 680, 0],
['formula', 668, 800, 0],
['synthesis', 668, 950, 0],
['CAE', 708, 115, 0], 
['others', 708, 223, 0],
]

answerMark = np.asarray(answer)
answerMark = np.asarray(answerMark[:,1:3], dtype=np.int)

judgeLine = 110

print(infile)
for i in range(0, answerMark.shape[0]):
    if resultFinal[answerMark[i,0], answerMark[i,1]] > judgeLine:
        answer[i][3] = 1
    print(answer[i][0], ':', answer[i][3])

