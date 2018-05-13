#!/usr/bin/python3
import os
import numpy as np
import csv
from pdf2image import convert_from_path
import cv2

def readPDF(infile, width, grayscale=True):
    """readPDF opens pdf as array to be hundled by opevCV. {{{
    When the pdf file has multiple pages, 
    automatically pick first page. 
    infile: a pdf file to open
    width: image width (pixel) of opened image
    grayscale = Ture : convert image to grayscale
    """ 

    #To open a pdf file.
    imgAllPages = convert_from_path(infile, dpi=100)
    img = imgAllPages[0]  #pick first page up
    img = np.asarray(img)
    img = img.take([1,2,0], axis=2)  #change color ch. (GBR -> RGB)
    
    #To scale image to designated width.
    if img.shape[1] != width:
        height = int(round(img.shape[0] / img.shape[1] * width))
        img = cv2.resize(img, (width, height), 
                         interpolation = cv2.INTER_CUBIC)

    #To convert image in grayscale. 
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img  
    #}}}

def correctMisalign(img, marker, center, compus, scope=100):
    """correctMisalign corrects misalignment/misscale of a image {{{
    by using two markers on the image. 
    First: To shift image according to center
    Second: To rescale and rotate according to the distance/angle between center&compus
    img: input image
    marker: marker image to fit (circle). 
            center of the cricle should be on the center of image. 
    center: coordinate of the center point [x, y]
    compus: coordinate of the compus point [x, y]
    scope: distance to search marker points from where it should be. 
    """

    markerCenter = np.asarray(marker.shape)//2
    guide = np.asarray([center, compus])
    landmark = np.zeros(guide.shape)
    
    #To run template matching to finder markers
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
        min_val, max_val, min_loc, landmark[i,:] = \
            cv2.minMaxLoc(np.multiply(resultPadded, mask))
    
    #To shift image
    shift = guide[0] - landmark[0] 
    M = np.float32([
    [1, 0, shift[0]] ,
    [0, 1, shift[1]] ])
    imgShifted = cv2.warpAffine(img, M, (width, height))
    
    #To rescale & rotate image
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
    the coordinate in answerList.
    img: input iamge
    marker: image of answer marker
    answerList = [['answer1', x, y, '0/1'], [...], ... ]
    threshold: threshold to judge whether markers are filled or not
    """

    markerCenter = np.asarray(marker.shape)//2
    width = img.shape[1]
    height = img.shape[0]

    #To run template matching to find answer markers
    resultFinal = cv2.matchTemplate(imgModified, marker, 0)
    resultFinal = (1-resultFinal/np.max(resultFinal))*255
    M = np.float32([
    [1, 0, markerCenter[1]] ,
    [0, 1, markerCenter[0]] ])
    resultFinal = cv2.warpAffine(resultFinal, M, (width, height))

    #To get coordinate of answer marker from answerList.
    answerCoord = np.asarray(answerList)
    answerCoord = np.asarray(answerCoord[:,1:3], dtype=np.int)
    
    #To judge each answer markers are filled or not.
    answers = []
    for i in range(0, answerCoord.shape[0]):
        if (resultFinal[answerCoord[i,1], answerCoord[i,0]] 
            > threshold):
            answers.append('1')
        else:
            answers.append('0')

    return answers
    #}}}

if __name__ == '__main__':
    """ __main__ runs readPDF/correctMisalign/checkAnswer seq 
    for all files in the folder 'scanPDF'. 
    """
    
    #Definition of image width and coordinate of center/compus
    width = 1084
    center = (31,  726)
    compus = (1051, 726)
    
    #Definition of answer list and their coordinates
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

    #To input marker image.
    marker= cv2.imread('marker.png', 0)

    #To get file list in the folder 'scanPDF'.
    files = os.listdir('./scanPDF')
    files_file = [f for f in files if os.path.isfile(os.path.join('./scanPDF', f))]

    output = ['filename']
    for i in range(0, len(answerList)):
        output.append(answerList[i][0])
    output = [output]

    for i in range(0, len(files_file)):
        infile = os.path.join('./scanPDF', files_file[i])
        img = readPDF(infile, width)
        
        width = img.shape[1]
        height = img.shape[0]
        
        imgModified = \
            correctMisalign(img, marker, center, compus, scope = 100)
        print('now checking file "', infile, '" ...', sep='')
        
        answer = [infile]
        answer.extend(checkAnswer(imgModified, marker, 
                                  answerList, threshold=110))
        output.append(answer)

    with open('result.txt', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(output)

    print('checkMark finished. Results are saved in result.txt.')
# vim: set foldmethod=marker :
