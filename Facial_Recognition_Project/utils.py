from random import choice, randint
from string import ascii_uppercase
from imutils import paths
import os
import face_recognition
import pickle
import cv2

## Generates a random string of X characters (defaults to 5)
def rand_string(length=5):
    return ''.join(choice(ascii_uppercase) for i in range(length))

def rand_num(maxdigits=5):
    endint = (10**maxdigits)-1
    return randint(0,endint)

def encoding(imgpath, wrencode='IST718.pickle', model='cnn'):
    ## Creating empty lists to store data during the loop
    knownEncodings = []
    knownNames = []

    imgs = list(paths.list_images(imgpath))
    #print(paths.list_images(imgpath))
    i = 0

    ### Iterating through the directory of images and encoding them
    for (i, imagePath) in enumerate(imgs):
        print("[INFO] processing image {}/{}".format(i + 1, len(imgs)))

        ## Getting the name from the file path
        name = os.path.dirname(imagePath).split("/")[1].replace("_", " ")

        ## Reading in the image and turning it into a format that opencv can use
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## Using the specified model to find the face. Defaults to cnn
        boxes = face_recognition.face_locations(rgb, model=model)

        ## Compute the encoding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            ## Appends the encoding and the name to the previously created empty lists
            knownEncodings.append(encoding)
            knownNames.append(name)



    print("[INFO] serializing encodings...")

    ## Storing the encoding and names in a dictionary
    data = {"encodings": knownEncodings, "names": knownNames}

    ## Writing the encoding to a file. Creates the file if it doesn't exist
    with open(wrencode, 'wb+') as f:
        f.write(pickle.dumps(data))



# Path to the training data stored with each person in their own folder
## Example: FaceData/Carlo_Mencarelli/image.png
imgpath = 'FaceData/'

# Where/what to save the image encoding as
wrencode = 'FaceEncoding/IST718_26training.pickle'


# Method to use
## cnn is: convolutional neural network - Accurate, but slow if not using CUDA
## hog is: histogram of oriented gradients - Fast, but less accurate
model = 'cnn'
#model = 'hog'

encoding(imgpath, wrencode, model)


def readconfig(configfile):
    pass