
#import needed libs
import cv2
from skimage import io, color
import sklearn
import pandas
import skimage
import matplotlib
from skimage.transform import resize
import matplotlib.pyplot as plt
import datetime
import os
import joblib
import schedule
import time
import sys


def get_parking_spots(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

mask = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\MASKOFPKLOTACROSSDREESE3.png'
parking_image_test = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\UPDATEDACROSSDREESE2.png'
parking_image_test1 = cv2.imread(parking_image_test)
parking_image_test2 = resize(parking_image_test1,(449,1908))


# # Load the image in color
# mask2 = cv2.imread(mask)

# # Convert the image to grayscale
# mask2_gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)


# cc = cv2.connectedComponentsWithStats(mask2_gray, 4, cv2.CV_32S)


# parking_spaces = get_parking_spots(cc)

mask2 = cv2.imread(mask,0)
# mask2 = resize(mask2,(1040,520))
# mask2 = cv2.convertScaleAbs(mask2)

cc = cv2.connectedComponentsWithStats(mask2, 4, cv2.CV_32S)

parking_spaces = get_parking_spots(cc)

print(parking_spaces[0])
# parking_spaces = get_parking_spots(cc)
open_space = 0
closed_space =0
frame = parking_image_test1
# frame = parking_image_test1
alpha = .3
# park_test = parking_spaces[:4]

# for space in park_test:
fdp2 = "C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\ACROSSDRESSE2"
i = 0
# parking_spaces = parking_spaces[:5]
for space in parking_spaces:
    x1, y1, w, h = space
    space_img = frame[y1:y1+h, x1+28:x1+w+38, :]

    # Define the file path for the cropped image
    file_name = f"cropd3{i}.png"
    fp = os.path.join(fdp2, file_name)
    # print("printing to acrossdreese2")
    # Save the cropped image
    cv2.imwrite(fp, space_img)
    # print("printed to acrossdreese2")

    i += 1