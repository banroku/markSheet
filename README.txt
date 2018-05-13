Module to automatically check answer marks on scanned bubble sheet. 
Usage: 



1. Make your original bubble sheet (for survey or test etc... 
The sheet needs following two features,
    1. Enbed two markers to fix misalignment and misscale 
       which might be induced on scanning the sheet. 
       The markers should be black circles 
       on two different corners (see example).
    2. Enbed blank circles for answer to let people to fill in. 
       The size should be same to marker for 1. 

2. Notes coordinates of markers
Use any image editor to open your sheets, 
and notes coordinates of two markers for correction 
(one for center[x, y] and one for compus[x, y]), 
and also the coordinates of all answer markers (for answerList).  

3. Scan your sheets into pdf 
Do your survey or test with your original bubble sheets, 
and scan them into pdf. 
Put pdf in the folder './scanPDF'. 

4. Prepare for running checkMark.py
checkMark.py including three parts, 
those are readPDF, correctMisalign, and checkAnswer. 
To use this program, firstly you should prepare following three. 
    1. marker.png
        Cut&paste your marker image into marker.png, 
        and put it in main folder. 
        marker center should be on the center of the image.

    2. fill coordinates of center and compus
        Open checkMark.py and change width, center, compus in __main__
        according to your own bubble sheet. 

    3. fill answerList
        Open checkMark.py and fill answerList 
        according to your own bubble sheet. 

5. Run checkMark.py by python3 
it output answers on your window. 



    readPDF (infile, width, grayscale=True)
        readPDF opens pdf as array to be hundled by opevCV.
        When the pdf file has multiple pages,
        it automatically picks first page.
    
    correctMisalign(img, marker, center, compus, scope=100)
        correctMisalign corrects misalignment/misscale of a image
        by using two markers on the image. 
        First: To shift image according to center
        Second: To rescale and rotate according to the distance/angle between center&compus
    
    checkAnswer(img, marker, answerList, threshold=110):
        checkAnswer checks answers according to 
        the coordinate in answerList.
        answerList = [['answer1', x, y, '0/1'], [...], ... ]
