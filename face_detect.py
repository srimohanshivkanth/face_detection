import cv2
import sys
import os
import imutils
from imutils.object_detection import non_max_suppression


imagePath = sys.argv[1]

cascPath = 'face.xml'
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
#faceCascade1 = cv2.CascadeClassifier(cascPath1)


# Read the image
image = cv2.imread(imagePath)

# Resize the image so it fits in the screen
image1 = imutils.resize(image,height=450)
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
  	#flags = cv2.cv.CV_HAAR_SCALE_IMAGE
  	flags=0
)

faces = non_max_suppression(faces,probs=None,overlapThresh=0.65)
if format(len(faces))==1:
	print("Found {0} face!".format(len(faces)))
else:
	print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow("Faces found", image1)
cv2.waitKey(0)
