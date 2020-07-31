import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='images'
#Create a list of all the images
images=[]
classNames=[]

myList=os.listdir(path)
print(myList)

#Using the above names to import the images
for cl in myList:
    cur_image=cv2.imread(f'{path}/{cl}')
    images.append(cur_image)
    classNames.append(os.path.splitext(cl)[0]) #Spltting name where we can seperate the first word and remove jpg
print(classNames) 

#Finding the encoding for each images
def findencodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


#Mark our attendence
def mark_attendence(name):
    with open('attendence.csv','r+') as f:
        #Dont want to repeat same names
        myDataList=f.readlines()
        nameList=[]
        print(myDataList)
        for line in myDataList:
            entry = line.split(',')    
            nameList.append(entry[0])
            
        if name not in nameList:
            now =datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
            


#Calling the function now
#Finding the encodings for the images known
encodeListKnown=findencodings(images)
print('Encoding complete')

#Matching image with  the existing image using webcam
cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    #Reduce the size of the image
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    #Encoding our images
    facescurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facescurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facescurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #Lowest distance will be the best match
        print(faceDis)
        #Find the lowest element
        matchIndex=np.argmin(faceDis)
        
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)  
            mark_attendence(name)          
            
    cv2.imshow("Webcam",img)
    cv2.waitKey(1)
        
        
# #Dtetcting the Face
# faceLoc=face_recognition.face_locations(imgElon)[0]
# encodeElon=face_recognition.face_encodings(imgElon)[0]

# #faceLoc return 4 values which is like Top,Bottom,Right,Left
# print(faceLoc)
# #Creating box on the image
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# faceLocTest=face_recognition.face_locations(imgTest)[0]
# encodeTest=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)        
        