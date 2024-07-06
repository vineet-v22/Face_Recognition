import cv2
import numpy as np
import face_recognition
import os
import sched,time
import csv
from datetime import datetime
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.keras.utils.disable_interactive_logging()


path = "./python/images"
images = []
classNames = []
persons = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    # Limit the name to 9 characters
    short_name = os.path.splitext(cl)[0][:9]
    classNames.append(short_name)


def findEncodings(images):
    encodeList = []
    for img in images:
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:  # Check if at least one face is detected
            encode = face_encodings[0]
            encodeList.append(encode)
        else:
            # Handle cases where no face is detected
            encodeList.append(None)
    return encodeList


def saveEncodings(encodeList, filename="./python/encodings.csv"):
    with open(filename, "w") as f:
        for encode in encodeList:
            f.write(",".join(map(str, encode)) + "\n")


# Function to load face encodings from a CSV file
def loadEncodings(filename="./python/encodings.csv"):
    encodeList = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()  # Remove leading/trailing whitespace
                if not line:
                    continue  # Skip empty lines
                values = line.split(",")
                encode = [float(e) for e in values if e]  # Skip empty values
                encodeList.append(encode)
    return encodeList


encodeListKnown = loadEncodings()
if not encodeListKnown:
    encodeListKnown = []

    encode = 0
    for img in images:
        if img is not None and img.size > 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:  # Check if at least one face is detected
                encode = face_encodings[0]
        encodeListKnown.append(encode)

        saveEncodings(encodeListKnown)

# Initialize the MTCNN model for face detection
detector = MTCNN()

from mtcnn import MTCNN  # Make sure you have MTCNN installed


def detect_and_extract_faces(input_image):
    # Load the input image

    # Convert the image to RGB format
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Use MTCNN to detect faces in the image
    detector = MTCNN()
    faces = detector.detect_faces(rgb_image)

    # Initialize an array to store detected face images
    detected_faces = []

    for i, face in enumerate(faces):
        x, y, width, height = face["box"]
        x1, y1, x2, y2 = (
            abs(x),
            abs(y),
            abs(x + width),
            abs(y + height),
        )  # Ensure positive values
        face_image = input_image[y1:y2, x1:x2]

        # Save each detected face, if any
        if face_image.size > 0:
            detected_faces.append(face_image)

    return detected_faces


path = "./python/image"

students={}
for i in range(1,83):
    if(i>=10):
      students['2200010'+str(i)]=0
    else:
        students['22000100'+str(i)]=0
students['220002018']=0
students['220002029']=0
students['220002063']=0
students['220002081']=0

def markAtt(name):
    students[name]=1

imageList = os.listdir(path)
for cl in imageList:
    input_image = cv2.imread(f"{path}/{cl}")
    detected_faces = detect_and_extract_faces(input_image)

    for i, face in enumerate(detected_faces):
        img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        imgS = cv2.resize(img, (0, 0), None, 1, 1)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        # print(encodeCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] and matchIndex < len(classNames):
                name = classNames[matchIndex]
                persons.append(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    img,
                    name,
                    (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                # cv2.imshow(name,img)

for person in persons:
    markAtt(person)
 
path = "./python/image/"
for name in os.listdir(path):
    name = path + name
    os.remove(name)

# Define the fieldnames for the CSV file
fieldnames = ["Roll No", "Attendance"]

# Specify the CSV file's name
csv_file = "./python/students_attendance.csv"

# Open the CSV file for writing
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row to the CSV file
    writer.writeheader()

    # Write the student data to the CSV file
    for roll_no, attendance in students.items():
        writer.writerow({"Roll No": roll_no, "Attendance": attendance})
print(students)

