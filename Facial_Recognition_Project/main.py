import cv2
import os
import time
import face_recognition
import pickle
from mss import mss
from PIL import Image
import pandas as pd
import argparse
import configparser


## Captures the current screen and returns the image ready to be saved
## Optional parameter to set incase there's more than 1 monitor.
## If the value set is outside of the valid range, set the value to 1
## Returns a raw image of the screen
def screen_cap(mnum=1):
    with mss() as sct:
        if mnum >= len(sct.monitors):
            mnum = 1
        monitor = sct.monitors[mnum]
        sct_img = sct.grab(monitor)
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

## Identifies faces and saves them into the imgdir directory
## Creates a temp dataframe with the Date, ElapsedSeconds, Name, and EngagementLevel
## imgfile: Image file with the faces that you want recognized.
## classdata: Date of the class
## secselapased: Number of seconds elapsed in the recording so far
## imgdir: Directory to save the individual images in
## picklefile: opened face recognition file
## Returns the temp dataframe
def cycle(imgfile, classdate, secselapsed, imgdir, picklefile, emotionpickle, saveimage):
    tempemotionframe = pd.DataFrame(columns=['Date', 'ElapsedSeconds', 'Name', 'EmotionScore', 'EyeCount'])
    img = cv2.imread(imgfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.025, minNeighbors=12, minSize=(50,50), flags=cv2.CASCADE_SCALE_IMAGE)
    #randletters = Project.utils.rand_string(randlength)
    for (x,y,w,h) in faces:
        sub_face = img[y:y + h + 2, x:x + w + 2]
        name = recognize(sub_face, picklefile, modl=model)
        if name is not "Unknown":
            emotion = emotionrec(sub_face, emotionpickle, modl=model)
            eyes = len(eye_cascade.detectMultiScale(sub_face))
            if eyes > 2:
                eyes = 2
            if saveimage == True:
                FaceFileName = imgdir + "/" + name + '_' + str(classdate) + "_" + str(secselapsed) + "_" + str(emotion) + "_" + str(eyes) + ".jpg"
                cv2.imwrite(FaceFileName, sub_face)
            tempemotionframe.loc[len(tempemotionframe)] = [classdate, secselapsed, name, emotion, eyes]
        else:
            pass
            #print("Skipping Unknown")
    return tempemotionframe

## Recognizes the person in the image
## rawimg: The raw image data that's captured after recognizing a face using CV
## pickledata: opened face recognition file
## modl: Which model to use. CNN or HOG. Defaults to CNN right now
def recognize(rawimg, emotionpickle, modl='cnn'):
    boxes = face_recognition.face_locations(rawimg, model=modl)
    encodings = face_recognition.face_encodings(rawimg, boxes)

    names = []

    for encode in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(emotionpickle["encodings"], encode)
        name = "Unknown"

        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = emotionpickle["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
    if len(names) > 0:
        name = names[0].replace(' ', '_').lower()
    else:
        name = 'Unknown'
    return name

## Takes in the raw image, emotion pickle file, and a model to use
## Reads the image and identifies an emotion.
## Assigns a score based on the emotion and returns the sum of the scores based on the identified emotion
def emotionrec(rawimg, pickledata, modl='cnn'):
    boxes = face_recognition.face_locations(rawimg, model=modl)
    encodings = face_recognition.face_encodings(rawimg, boxes)

    scores = []

    for encode in encodings:
        matches = face_recognition.compare_faces(pickledata["encodings"], encode)
        score = 0
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            for i in matchedIdxs:
                name = pickledata["names"][i]
                if name == 'positive':
                    score = 2
                elif name == 'negative':
                    score = -0.5
                else:
                    score = 0
                scores.append(score)
    if len(scores) > 0:
        return sum(scores)
    else:
        return 0

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--configfile', help='Config file')
    args = vars(argparser.parse_args())
    configfile = args['configfile']
    ### --configfile=C:\git\IST718\Project\base.config

    ## Reading from a config file
    config = configparser.ConfigParser()
    config.read(configfile)
    allconf = config['BASE']

    wait = allconf.get('wait')
    run_time = allconf.get('run_time')
    face_file = allconf.get('face_file').replace("'", "")
    eye_file = allconf.get('eye_file').replace("'", "")
    classdate = allconf.get('classdate').replace("'", "")
    monnum = int(allconf.get('monnum'))
    model = allconf.get('model')
    face_rec = allconf.get('face_rec').replace("'", "")
    emotion_rec = allconf.get('emotion_rec').replace("'", "")
    save_img = allconf.get('save_img').replace("'", "")
    save_dir = allconf.get('save_dir').replace("'", "")
    df_path = allconf.get('df_path').replace("'", "")
    df_export = allconf.get('df_export').replace("'", "")


    ## Checking some values and correcting if needed
    ### If the wait time isn't a valid integer or is too small or too big, correct it
    try:
        wait = int(wait)
        if wait < 5:
            wait = 5
        elif wait > 900:
            wait = 900
    except:
        wait = 15

    ### Checking to make sure the monitor value is a valid integer
    try:
        monnum = int(monnum)
    except:
        monnum = 1

    ### If the run time isn't a valid integer or is too small or too big, correct it
    try:
        run_time = int(run_time)
        if run_time < wait:
            run_time = wait + 1
        elif run_time > 6000:
            run_time = 5400
    except:
        run_time = 5400

    ### If save_img is not a boolean value make it one. Defaults the variable to false if needed
    if save_img.lower() == 'true':
        save_img = True
    elif save_img.lower() == 'false':
        save_img = False
    else:
        save_img = False

    ### If df_export is not a boolean value make it one. Defaults the variable to false if needed
    if df_export.lower() == 'true':
        df_export = True
    elif df_export.lower() == 'false':
        df_export = False
    else:
        df_export = False

    ### If the model specified isn't HOG or CNN, set it to HOG
    if not (model.lower() == 'cnn' or model.lower() == 'hog'):
        model = 'hog'

    ## Setting some other random variables
    ### How many loops to run
    max_iterations = round(run_time / wait, 0)
    ### The current dir of the script being run
    currentdir = os.path.dirname(os.path.abspath(__file__))

    ## Initializing the empty dataframe
    emotionframe = pd.DataFrame(columns=['Date', 'ElapsedSeconds', 'Name', 'EmotionScore', 'EyeCount'])

    ## Checking various directories and files
    ### Check for the output directory for the images and the CSV
    if save_img is True:
        try:
            os.makedirs(save_dir)
            print('{0} created'.format(save_dir))
        except OSError:
            print('{0} already exists'.format(save_dir))

    if df_export is True:
        try:
            os.makedirs(os.path.dirname(df_path))
            print('{0} created'.format(df_path))
        except OSError:
            print('{0} already exists'.format(df_path))

    ### Import the cascade files for face detection
    if os.path.isfile(face_file):
        face_cascade = cv2.CascadeClassifier(face_file)
        print('Importing cascade file {}.'.format(face_file))
    else:
        exit('Error! {} doesn\'t exist.'.format(face_file))

    ### Import the cascade files for eye detection
    if os.path.isfile(eye_file):
        eye_cascade = cv2.CascadeClassifier(eye_file)
        print('Importing cascade file {}.'.format(eye_file))
    else:
        exit('Error! {} doesn\'t exist.'.format(eye_file))

    ### Loading the facial recognition encoding
    if os.path.isfile(face_rec):
        data = pickle.loads(open(face_rec, 'rb').read())
        print('Importing facial recognition file {}'.format(face_rec))
    else:
        exit('Error! {} doesn\'t exist.'.format(face_rec))

    ### Loading the emotion recognition encoding
    if os.path.isfile(emotion_rec):
        emotiondata = pickle.loads(open(emotion_rec, 'rb').read())
        print('Importing facial recognition file {}'.format(emotion_rec))
    else:
        exit('Error! {} doesn\'t exist.'.format(emotion_rec))


    ## Starting timer
    start = time.time()
    finish = start + run_time

    ## While the current time is still less then the projected end time
    while time.time() < finish:
        ## Calculating the elapsed time so far
        elapsed = round(time.time() - start, 0)

        ## Clearing the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')

        ## Capturing the current screen on the monitor and saving it as temp.jpg
        tempimg = screen_cap(mnum=monnum).save('temp.jpg')

        ## Running through the various recognitions and returns the data as a Pandas Dataframe
        count = cycle(imgfile='temp.jpg', classdate=classdate, secselapsed=elapsed, imgdir=save_dir, picklefile=data,
                      emotionpickle=emotiondata, saveimage=True)

        ## Print the elapsed time, max time, save directory, export directory, and the temp dataframe
        print('Configured Run Time: {} seconds'.format(run_time))
        print('Elapsed Time: {} seconds'.format(elapsed))
        if save_img is True:
            print('Saving Images to: {}'.format(save_dir))
        if df_export is True:
            print('Dataframe will be saved to: {}'.format(df_path))
        print('\n')
        print(count)

        ## Concat the temp df to the perm dataframe
        emotionframe = pd.concat([emotionframe, count], ignore_index=True)

        ## When using the CNN model with CPU there's more procesing time which means the capture eventually beings to
        ## lag behind. The wait timer may need to be adjusted
        ## If using GPU and CNN model or just the HOG model then there's no need to wait extra or less time
        time.sleep(wait)


    end = time.time()
    total = end - start

    print('Runtime: {} seconds'.format(total))

    if df_export is True:
        print('Exporting dataframe: {}'.format(df_path))
        emotionframe.to_csv(df_path)
