#Required Packages and Library
import face_recognition as fr
import cv2
import os

#Capturing video
img = cv2.VideoCapture(0)
model = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = []
for imgs in os.listdir('Face Models'):
    faces.append(imgs)
encoded = []

#Fetching Face Models with thier name
for name in faces:
    image =fr.load_image_file('Face Models\\'+name, mode='RGB')
    encoded.append(fr.face_encodings(image)[0])

#Comparing Captured vedio with Face Models 
while True:

    try:
        r,i = img.read()
        face = model.detectMultiScale(i)

        if len(face) == 0:print("No Face Detected")

        elif len(face) > 0:
            for (x, y, s1, s2) in face:
                crop = i[y-25 : y+s1+25, x-25 : x+s2+25]
                face_enc = fr.face_encodings(crop)[0]
                result = fr.compare_faces(encoded, face_enc)
                for val in result:
                    if val == True:ind = result.index(val)
                
                #Making Square Box with Thier Name
                cv2.rectangle(i, (x, y), (x+s1,y+s2), (0,165,255), 3)
                cv2.putText(i,faces[ind][:-4],(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
            
            cv2.imshow('Video',i)

    except Exception as e:
        print("Please Show your Face Properly")
    
    #To Break the while loop and to close the window press Escape Button  
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows
