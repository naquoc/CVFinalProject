import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
# return list of all folder in source_emotion
participants = glob.glob("source_emotion\\*")

for x in participants:
    #get last 4 character in path of emotion
    part = "{}".format(x[-4:])
    for sessions in glob.glob("%s\\*" %x):
        for files in glob.glob("%s\\*" %sessions):
            current_session = files[20:-30]
            file = open(files, 'r')
            emotion = int(float(file.readline()))
            #get path for last image in sequence, which contains the emotion
            sourcefile_emotion = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[-1]
            #do same for neutral image 
            sourcefile_neutral = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[0]
            
            #Generate path to put neutral image
            dest_neut = "sorted_set\\neutral\\%s" %sourcefile_neutral[24:]
            #Do same for emotion containing image 
            dest_emot = "sorted_set\\%s\\%s" %(emotions[emotion], sourcefile_emotion[24:])
            
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file