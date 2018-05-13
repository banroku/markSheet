Module to automatically check answer marks on scanned bubble sheet. 
Usage: 

1. Make your original bubble sheet (for survey or test etc... 
The sheet needs following two features,
(see example "sample-bubblesheet.pdf")

    1. Enbed two markers to fix misalignment/misscale on scanning.
       The markers should be black circles on two corners 

    2. Enbed blank circles for answer to let people to fill in. 
       The size should be same to marker of 1. 

2. Notes coordinates of markers
Use any image editor to open your sheets, 
note coordinates of two markers for misalign correction 
(one for center[x, y] and one for compus[x, y]), 
and also note the coordinates of all answer markers (for answerList).  
3. Scan your sheets into pdf 
Print out your original bubble sheet and do your survey/test.  
Scan recovered sheets into pdf and put them in folder './scanPDF'. 

4. Prepare for running checkMark.py
checkMark.py including three parts, 
readPDF, correctMisalign, and checkAnswer. 
To use this program, firstly you should prepare following three. 

    1. Modify coordinates of center and compus, and width 
        Open checkMark.py and change width, center, compus in __main__
        according to your own bubble sheet. 

    2. Modify answerList
        Open checkMark.py and modify answerList 
        according to your own bubble sheet. 

    3. Prepare marker.png
        Cut&paste your marker image into marker.png, 
        and put it in main folder. 
        marker center should be on the center of the image.
        (see sample marker.png)

5. Run checkMark.py by python3 
checkMark checks answers on each pdf in 'scanPDF/', 
and save result in the result.txt. 



Reference: 

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
