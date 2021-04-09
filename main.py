import cv2
import numpy as np
import time

# Everybody wants magic and here I am with one! Be a Harry potter and make invisible yourself!

# Capturing the video
cap = cv2.VideoCapture(0)



time.sleep(3)

background = 0

# Capturing the background
for i in range(30):
    ret, background = cap.read()

background = np.flip(background, axis=1)

while cap.isOpened():
    ret, img = cap.read()

    # Flip the image
    img = np.flip(img, axis=1)

    # Convert the  , ge to HSV color space. It is done so to detect red colour
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # IN BGR combination [0 0 255] means Blue=0 Green =0 Red=255
    red = np.uint8([[[0, 0, 255]]])

    hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

    blurred = cv2.GaussianBlur(hsv, (35, 35), 0)

    # Defining lower range for red color detection.Done so that different red colors can be detected

    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])

    # Mask is created to capture red things
    mask_1 = cv2.inRange(hsv, lower, upper)



    # cv2.imshow('Display', masked_1)
    # Defining upper range for red color detection
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask_2 = cv2.inRange(hsv, lower_red, upper_red)

    mask_1 = mask_1 + mask_2
    # Addition of the two masks to generate the final mask.

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    # Create inverse of mask to show the things which were not black
    mask_2 = cv2.bitwise_not(mask_1)

    # Mask the background with Red things and pass it through red
    part1 = cv2.bitwise_and(background, background, mask=mask_1)

    # Shows all things which are not red
    part2 = cv2.bitwise_and(img, img, mask=mask_2)

    masked = cv2.addWeighted(part1, 1, part2, 1, 0)

    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Replacing pixels corresponding to cloak with the background pixels.
    masked[np.where(masked == 255)] = background[np.where(masked == 255)]
    cv2.imshow('Display', masked)
    if cv2.waitKey(10) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
