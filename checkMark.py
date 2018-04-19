# checkMark.py checks marks on the answer sheet. 

import numpy as np
import cv2

landmark = np.zeros((3,2))

img = cv2.imread('sample-image.png', 0)
marker= cv2.imread('landmark.png', 0)
markerCenter = [20, 21]
result = cv2.matchTemplate(img, marker, 0)

answer = [
['molding', 668, 115, 0],
['processing', 668, 241, 0],
['evaluation', 668, 374, 0],
['data-analysis', 668, 508, 0], 
['discussion', 668, 676, 0],
['formula', 668, 802, 0],
['synthesis', 668, 943, 0],
['CAE', 708, 115, 0], 
['others', 708, 241, 0],
]

#! for でanswerの座標から、markerCenterを引き算する。



#位置決め
maskAll = np.ones(result.shape) * np.max(result)
mask = maskAll[:]
mask[:49, :49] = 0
resultMasked = result + mask
min_val, max_val, landmark[0,:], max_loc = cv2.minMaxLoc(resultMasked)

mask = maskAll[:]
mask[-49:, :49] = 0
resultMasked = result + mask
min_val, max_val, landmark[1,:], max_loc = cv2.minMaxLoc(resultMasked)

mask = maskAll[:]
mask[-49:, -49:] = 0
resultMasked = result + mask
min_val, max_val, landmark[2,:], max_loc = cv2.minMaxLoc(resultMasked)


#calculate origin
#calculate slope of x, y

#define coordinates of answer marks (dictで作る, これが面倒くさい。
#あるいは、全部黒塗りにされているシートから自動で読み込む。
#そもそもマークシートの形状は円形いったくだから形状の工夫の余地はない

#convert coordinates of answer marks along origin/slopex, slope y

#judge answer 
 
#makerを色々変えられるようにしてみる。
#gimpでmakerを変えて、精度がよくなるようなマーカーを探す
