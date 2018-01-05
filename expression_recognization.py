import cv2
import glob
import random
import numpy as np

emotions = ["neutral", "anger", "disgust", "happy", "surprise"] #Emotion list
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 95/5
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.95)] #get first 95% of file list
    prediction = files[-int(len(files)*0.05):] #get last 5% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def run_expression_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
   
    print ("training fisher face classifier")
    print ("size of training set is:", len(training_labels), "images")
    # train base on training_set
    fishface.train(training_data, np.asarray(training_labels))

    print ("predicting classification set")
    index = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        # predict emotion
        pred, conf = fishface.predict(image)
        # if predicted emotion equal emotion_labels image has
        if pred == prediction_labels[index]:
            # increament prediction correct
            correct += 1
            index += 1
        else:
            incorrect += 1
            index += 1
    return ((100*correct)/(correct + incorrect))

if __name__ == "__main__":
    results = []
    #run 5 times
    for i in range(0,5):
        correct = run_expression_recognizer()
        print ("got", correct, "percent correct!")
        results.append(correct)

    print ("\n\nend performance: ", np.mean(results), " percent correct!")