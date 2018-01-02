import cv2
import sys
import glob
faceDet = cv2.CascadeClassifier("harr-classifier/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("harr-classifier/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("harr-classifier/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("harr-classifier/haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

def detect_faces(emotion):
    files = glob.glob("sorted_set\\%s\\*" %emotion) #Get list of all images with emotion
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        # preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        # gray = cv2.GaussianBlur(gray,(5,5),0) #Blur
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
            print ("face found in file: %s" %f)
        #    rectangle face region
        #    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number

for emotion in emotions:
    detect_faces(emotion)