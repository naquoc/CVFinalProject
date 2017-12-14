import cv2
import sys

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        image = cv2.imread(str(sys.argv[1]))
        cv2.imshow("Image", image)
        # preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        gray = cv2.GaussianBlur(gray,(5,5),0) #Blur
        # detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10  , minSize=(5, 5))
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5))
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5))
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5))
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        for (x,y,w,h) in facefeatures:
        #    rectangle face region
        #    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y+h, x:x+w]
            output = cv2.resize(face, (96, 96))
            cv2.imshow('Face', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()