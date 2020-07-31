import cv2
import numpy as np
import face_recognition


#Loading image
imgElon=face_recognition.load_image_file("images/elon_mask.jpg")
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file("images/Billgates.jpg")
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#Dtetcting the Face
faceLoc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]

#faceLoc return 4 values which is like Top,Bottom,Right,Left
print(faceLoc)
#Creating box on the image
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#Next step comparing the faces and finding the distance between them
#Comparing the encodings
results=face_recognition.compare_faces([encodeElon],encodeTest)
#To find the best results we will use face distance
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
print(faceDis)
print(results)

cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)



cv2.imshow("Image1",imgElon)
cv2.imshow("Image",imgTest)

cv2.waitKey(0)