import cv2
import numpy as np
import time
#Everybody wants magic and here I am with one! Be a Harry potter and make invisible yourself!
cap = cv2.VideoCapture(0)
time.sleep(3)
background=0

for i in range(60):
    ret,background = cap.read()

background = np.flip(background,axis=1)

while(cap.isOpened()):
    ret, img = cap.read()

    # Flip the image
    img = np.flip(img, axis = 1)

    # Convert the image to HSV color space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (35, 35), 0)

    # Defining lower range for red color detection.
    lower = np.array([0,120,70])
    upper = np.array([10,255,255])
    masked_1 = cv2.inRange(hsv, lower, upper)

    # Defining upper range for red color detection
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    masked_2 = cv2.inRange(hsv, lower_red, upper_red)

    # Addition of the two masks to generate the final mask.
    masked = masked_1 + masked_2
    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # Replacing pixels corresponding to cloak with the background pixels.
    img[np.where(masked == 255)] = background[np.where(masked == 255)]
    cv2.imshow('Display',img)
    k = cv2.waitKey(10)
    if k == 27:
        break